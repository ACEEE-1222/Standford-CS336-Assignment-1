import os
from typing import BinaryIO

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
    # token_len = len(split_special_token)
    # prev_tail = b""  # 上一次 mini_chunk 的末尾部分，用于拼接

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
            # 拼接上次尾部，保证不会漏掉跨块的 token
            # mini_chunk = prev_tail + mini_chunk

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 注意：found_at 是相对于 mini_chunk 的偏移
                # 实际位置需减去 prev_tail 长度
                # true_position = initial_position - len(prev_tail) + found_at
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            # 更新 prev_tail，为下一次拼接准备
            # prev_tail = mini_chunk[-(token_len - 1):]
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

## Usage
with open(..., "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8"))
        
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token