import os
import regex as re
from typing import BinaryIO
from typing import Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue
import time

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        parts = re.split('(' + pattern + ')', text)

    return parts

def pretokenize(text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
    """
    Seperating text into pretokens
    Special tokens are independent pretokens
    """
    parts = split_by_special_tokens(text, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:  # Keep special tokens, otherwise ignore
                spec_tok_bytes = part.encode('utf-8')
                tokens_list.append([spec_tok_bytes])
        else:
            str_tokens = re.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in str_tokens]
            tokens_list.append(part_tokens)
    tokens = [token for part_tokens in tokens_list for token in part_tokens]
    return tokens

def worker(text: str, special_tokens: list[str], q: Queue):
    """Worker pretokenizing process for multiprocessing"""
    pretokens = pretokenize(text, special_tokens)
    q.put(pretokens)
    # print("done")

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    merges = []

    # Chunk the text file
    num_processes = 4
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # Parallelizing pretokenization
    pretokens_list = []
    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pretokens = [token for tokens in pretokens_list for token in tokens]

    # Merging
    counts = defaultdict(int)
    index_dict = defaultdict(set)  # Store pretoken location for each pair

    for j, pretoken in enumerate(pretokens):
        for index1, index2 in zip(pretoken, pretoken[1:]):
            counts[index1, index2] += 1
            index_dict[index1, index2].add(j)

    for i in range(num_merges):
        # Prefer lexicographically greater pair
        # Example: max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) = ('BA', 'A')
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]

        index1, index2 = max_pair

        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))

        merge(counts, index_dict, pretokens, max_pair, new_index)

    return (vocab, merges)

def merge(counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int],set[int]], pretokens: list[list[int]], max_pair: (int, int), new_index: int):
    """Merge the pairs with highest frequency and update counts, index_dict"""
    index_set = index_dict[max_pair]

    for i in index_set:
        pretoken = pretokens[i]
        new_pretoken = []

        pos_list = []   # Store positions of max_pair for each new pretoken after merge
        pos = 0
        j = 0

        # Replace max_pair with new_index in each pretoken
        while j < len(pretoken):
            if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == max_pair):
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        # Update counts and index_dict
        for pos in pos_list:
            counts[max_pair] -= 1

            if pos > 0:
                if new_pretoken[pos-1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1    
                else:
                    counts[(new_pretoken[pos-1], max_pair[0])] -= 1

                counts[(new_pretoken[pos-1], new_pretoken[pos])] += 1
                index_dict[(new_pretoken[pos-1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken)-1:
                if new_pretoken[pos+1] == new_index:
                    counts[(max_pair[1], max_pair[0])] -= 1     
                else:
                    counts[(max_pair[1], new_pretoken[pos+1])] -= 1

                counts[(new_pretoken[pos], new_pretoken[pos+1])] += 1
                index_dict[(new_pretoken[pos], new_pretoken[pos+1])].add(i)

        pretokens[i] = new_pretoken


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges"""
        raise NotImplementedError

    def encode(self, text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        byte_pretokens = pretokenize(text, self.special_tokens, drop_special_token=False)   # list[bytes]
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = []  # list[list[int]]

        # Convert pretokens from bytes to list[int] by vocab
        for i, pretoken in enumerate(byte_pretokens):

            new_pretoken = []

            if pretoken in byte_special_tokens:
                index = vocab_reversed[pretoken]
                new_pretoken.append(index)
            else:
                for b in pretoken:
                    index = vocab_reversed[bytes([b])]
                    new_pretoken.append(index)

            pretokens.append(new_pretoken)

        # Merge
        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_pretoken = []
                new_index = vocab_reversed[merge[0] + merge[1]]
                j = 0
                while j < len(pretoken):
                    if (j < len(pretoken)-1) and ((self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1

                pretoken = new_pretoken

            pretokens[i] = pretoken

        tokens = [token for pretoken in pretokens for token in pretoken] 
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files 
        that we cannot directly load into memory.
        """
        for line in iterable:
            for idx in self.encode(line):
                yield idx


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]    # bytes
            else:
                token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 


def main():
    file_path = "./data/corpus.en"
    vocab_size = 500
    # special_tokens = ["<|endoftext|>"]
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    # print(merges)

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    encoded = tokenizer.encode(test_string)
    print("encoded:",encoded)
    decoded = [tokenizer.decode([x]) for x in encoded]
    print("decoded:", decoded)

    print(test_string == decoded)

    # print(vocab)

def test():
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|><|endoftext|>", "<|endoftext|>"})
    decoded = [tokenizer.decode([x]) for x in ids]
    print(decoded)

if __name__ == "__main__":
    main()
    # test()