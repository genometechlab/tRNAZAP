import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, dim))

    def forward(self, T):
        return self.pe[:T].unsqueeze(0)  # shape: [1, T, D]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T].unsqueeze(0)  # shape: [1, T, D]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.relative_embedding = nn.Parameter(torch.randn(2 * max_len - 1, dim))

    def forward(self, T):
        # returns relative position bias matrix: [T, T, D]
        pos_indices = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)
        pos_indices = pos_indices.clamp(-self.max_len + 1, self.max_len - 1) + self.max_len - 1
        return self.relative_embedding[pos_indices]  # shape: [T, T, D]