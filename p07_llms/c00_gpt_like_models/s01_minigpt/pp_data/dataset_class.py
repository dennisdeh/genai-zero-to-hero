import torch
from torch.utils.data import Dataset


# Define a custom dataset class appropriate for our task
class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def get_vocab_size(self):
        return 100277  # cl100k_base vocab size

    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        # We grab a chunk of tokens of length block_size + 1
        # x is the sequence, y is the sequence shifted by 1 (the targets)
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
