import os
import re

import torch
from torch.utils.data import Dataset


class UnicodeData(Dataset):
    def __init__(self, data_dir, block_size=128):
        self.data = set()

        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                parts = re.split(r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s])', f.read())
                self.data.update(parts)

        self.data = sorted(list(self.data))
        self.block_size = block_size

        self.vocab_size = len(self.data)
        print(f"Vocab size: {self.vocab_size}, block size: {self.block_size}")

        self.encode = {ch: i for i, ch in enumerate(self.data)}
        self.decode = {i: ch for i, ch in enumerate(self.data)}

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def encode_sequence(self, sequence):
        return [self.encode[s] for s in sequence]

    def decode_sequence(self, sequence):
        return "".join([self.decode[s] for s in sequence])

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every token to an integer
        dix = [self.encode[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y