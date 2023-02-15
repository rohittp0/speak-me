import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os.path import join
from typing import Generator, List, Tuple

from emoji import demojize

VALID_RANGES = [(32, 126), (3330, 3439)]

comp_type = List[Tuple[int, int, int]]


def get_compacted_ranges() -> Tuple[comp_type, int]:
    range_intervals = [e - s for s, e in VALID_RANGES]
    range_intervals.insert(0, 0)

    compacted = []
    for i, (s, e) in enumerate(VALID_RANGES):
        compacted.append((s, e, range_intervals[i] - s))

    return compacted, sum(range_intervals)


def get_encoded(num: int, ranges: comp_type) -> int:
    for s, e, offset in ranges:
        if s <= num <= e:
            return num + offset

    return -1


def get_files(data_dir) -> Generator[str, None, None]:
    for cpath, folders, files in os.walk(data_dir):
        for file in files:
            yield join(cpath, file)


def process_chunk(compacted, file_root, file_index, chunk):
    encoded_chars = []
    for line in chunk:
        for char in demojize(line):
            encoded = get_encoded(ord(char), compacted)
            if encoded != -1:
                encoded_chars.append(encoded)
    with open(f"{file_root}/tokens-{file_index}.txt", "w") as f:
        f.write(",".join(str(x) for x in encoded_chars) + ",")
    return file_index


def run(data_dir: str):
    compacted, total = get_compacted_ranges()

    file_root = "tokens"
    os.makedirs(file_root, exist_ok=True)

    process_chunk_partial = partial(process_chunk, compacted, file_root)

    with ThreadPoolExecutor(max_workers=4) as executor:  # adjust max_workers as needed
        futures = []
        file_index = 0

        for file in get_files(data_dir):
            with open(file, "r", encoding="utf-8") as f2:
                chunk = []
                for line in f2:
                    chunk.append(line)
                    if len(chunk) >= 1000:  # adjust chunk size as needed
                        futures.append(executor.submit(process_chunk_partial, file_index, chunk))
                        file_index += 1
                        chunk = []
                if chunk:
                    futures.append(executor.submit(process_chunk_partial, file_index, chunk))
                    file_index += 1

        for future in futures:
            file_index = future.result()

        with open("tokens.txt", "w") as f:
            for i in range(file_index):
                with open(f"{file_root}/tokens-{i}.txt", "r") as f2:
                    tokens = f2.read()
                    f.write(tokens)


if __name__ == "__main__":
    run("data")
