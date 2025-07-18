import json
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict
import time

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initialize the Tokenizer with a vocabulary, BPE merge rules, and optional special tokens.

        Args:
            vocab (dict[int, bytes]): A mapping from token IDs to byte-encoded tokens.
            merges (list[tuple[bytes, bytes]]): A list of merge operations as tuples of byte pairs.
            special_tokens (list[str] | None): Optional list of user-defined special tokens to include.
        """
        self.vocab = vocab
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath (str): Path to the vocabulary file (from BPE training).
            merges_filepath (str): Path to the merges file (from BPE training).
            special_tokens (list[str] | None): Optional list of special tokens to include.

        Returns:
            Tokenizer: A Tokenizer instance initialized with the given files.
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = token_str.encode("utf-8")  # store as bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs using the BPE algorithm.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        # byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        token_ids = []
        pre_tokens_list = process_chunk((text, self.special_tokens, True))
        for tokens in pre_tokens_list:
            for pair in self.merges:
                a, b = pair
                new_tok = a + b
                new_tokens: list[bytes] = []
                i=0
                while i < len(tokens):
                    if i < len(tokens) -1 and (tokens[i],tokens[i+1]) == pair:
                        new_tokens.append(new_tok)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            for i in range(len(tokens)):
                token_ids.append(self.vocab_reversed.get(tokens[i]))
        
        return token_ids


    def encode_iterable(self, iterable: list[str]) -> iter:
        """
        Lazily encode an iterable of strings into a stream of token IDs.

        Useful for memory-efficient tokenization of large datasets.

        Args:
            iterable (list[str]): An iterable of strings (e.g., lines from a file).

        Returns:
            iter: A generator that yields token IDs one at a time.
        """
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]  
            else:
                token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE (Byte Pair Encoding) tokenizer on the given input text file.

    Parameters
    ----------
    input_path : str
        Path to a UTF-8 encoded text file containing training data for the BPE tokenizer.
        Each line is considered part of the corpus.

    vocab_size : int
        The total size of the final vocabulary (must include initial byte-level tokens,
        all merged tokens produced during training, and the given special tokens).

    special_tokens : list[str]
        A list of user-defined special tokens (e.g., ["<|endoftext|>", "<pad>"]) to be 
        added to the vocabulary. These tokens do NOT participate in merge decisions.

    num_processes : int, optional (default=8)
        Number of parallel processes used during pre-tokenization. Each process handles
        a chunk of the input corpus split at special token boundaries. More processes
        generally mean faster pre-tokenization.

    Returns
    -------
    vocab : dict[int, bytes]
        A dictionary mapping token IDs (integers) to token values (in bytes). The token 
        IDs should be assigned sequentially starting from 0.

    merges : list[tuple[bytes, bytes]]
        A list of BPE merge operations, where each tuple represents two byte-level tokens 
        that were merged together. The list should be ordered by merge time (first merge first).
    """

    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))

    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
        chunk_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)
    task_args = [(chunk, special_tokens, False) for chunk in chunk_list]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 3. Compute BPE merges
    merges : list[tuple[bytes, bytes]] = []
    pre_tokens_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break
            
        max_pair: tuple[bytes, bytes] = None
        max_cnt= -1
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i+1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)

            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j]=merged

            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges

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
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def find_chunk_boundaries_in_bytes(
    text: bytes,
    desired_num_chunks: int,
    split_special_token: str
) -> list[int]:
    """
    Chunk the input string into parts that can be processed independently.
    May return fewer chunks if boundaries overlap or cannot be found exactly.

    Args:
        text_bytes (bytes): Input byte sequence to be chunked.
        desired_num_chunks (int): Desired number of chunks.
        split_special_token (str): Special token string to split on (e.g., "\n").

    Returns:
        list[int]: Sorted list of chunk boundary indices in the string.
    """
    text_len = len(text)
    chunk_size = text_len // desired_num_chunks

    # Initial equally spaced boundaries (string indices)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = text_len  # Ensure last boundary is at end

    mini_chunk_size = 256  # number of characters to look ahead

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_pos = chunk_boundaries[bi]
        # Look ahead from initial_pos to find split_special_token
        search_pos = initial_pos

        while search_pos < text_len:
            # Extract a mini chunk to search in
            mini_chunk = text[search_pos:search_pos + mini_chunk_size]

            # Find special token in mini_chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Adjust boundary to position of token found
                true_pos = search_pos + found_at
                chunk_boundaries[bi] = true_pos
                break

            # If token not found in mini_chunk, advance by mini_chunk_size
            search_pos += mini_chunk_size
        else:
            # If we reached end of text without finding token, set boundary to text_len
            chunk_boundaries[bi] = text_len

    # Remove duplicates, sort, and return
    return sorted(set(chunk_boundaries))




def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    chunk, special_tokens, keep_special_tokens = args
    """
    Processes a chunk of text and returns byte pair frequency counts.

    Args:
        chunk (str): A chunk of text data (already decoded).
        special_tokens (list[str]): List of special tokens that should not be merged across.
        keep_special_tokens (bool): Whether to preserve special tokens as standalone tokens.

    Returns:
        pre_token_bytes (list[list[bytes]]): list of tokens, where each token is a list of bytes
    """
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"

    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        # if not segment:
        #     continue
        if keep_special_tokens and segment in special_tokens:
            # Treat the whole special token as a single token
            token_bytes = [segment.encode("utf-8")]
            pre_tokens_bytes.append(token_bytes)
        else:
            # Standard tokenization
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes


def save_bpe_model(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Save vocab
    vocab_serialized = {str(i): token.decode('utf-8', errors='replace') for i, token in vocab.items()}
    with open(os.path.join(output_dir, "TinyStories_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

    # Save merges
    with open(os.path.join(output_dir, "TinyStories_merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            a_str = a.decode("utf-8", errors="replace")
            b_str = b.decode("utf-8", errors="replace")
            f.write(f"{a_str} {b_str}\n")
    
def main():
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")
    print(f"Vocab size: {len(vocab)}")
    print(f"Longest token: {max(vocab.values(), key=len)} (length={len(max(vocab.values(), key=len))})")
    save_bpe_model(vocab, merges, "cs336_basics")

def test():
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|><|endoftext|>", "<|endoftext|>"})
    decoded = [tokenizer.decode([x]) for x in ids]
    print(f"tiktoken encoded: {ids}, decoded: {decoded}")

if __name__ == "__main__":
    # main()
    test()

