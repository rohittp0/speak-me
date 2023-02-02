import os
import pickle
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import Dataset


class UnicodeData(Dataset):
    def __init__(self, data_dir, block_size=128):
        self.data = self.get_tokens(data_dir)
        self.encode, self.decode = self.get_encodings()

        self.vocab_size = self.decode.shape[0]
        self.block_size = block_size

        print(f"Vocab size: {self.vocab_size}")

    @staticmethod
    def get_file_list(data_dir) -> Generator[Path, None, None]:
        for cpath, folders, files in os.walk(data_dir):
            for file in files:
                yield Path(cpath, file)

    def get_tokens(self, data_dir) -> np.ndarray:
        if os.path.exists("tokens.npy"):
            return np.load("tokens.npy", mmap_mode="r", encoding="bytes")

        file_list = self.get_file_list(data_dir)
        tokens = []

        for file in file_list:
            tokens += list(file.read_text(encoding="utf-8"))

        tokens = np.array(tokens)
        np.save("tokens.npy", tokens)

        return tokens

    def get_encodings(self):
        if os.path.exists("encode.pkl") and os.path.exists("decode.npy"):
            return (pickle.load(open("encode.pkl", "rb")),
                    np.load("decode.npy", mmap_mode="r", encoding="bytes"))

        tokens = np.unique(self.data)
        encode = {ch: i for i, ch in enumerate(tokens)}
        decode = tokens

        pickle.dump(encode, open("encode.pkl", "wb"))
        np.save("decode.npy", decode)

        return encode, decode

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def encode_sequence(self, sequence):
        return [self.encode[s] for s in sequence]

    def decode_sequence(self, sequence):
        return "".join([self.decode[s] for s in sequence])

    def __len__(self):
        return self.data.shape[0] - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every token to an integer
        dix = [self.encode[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    UnicodeData("data", 1024)
