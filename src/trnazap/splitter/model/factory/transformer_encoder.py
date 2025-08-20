import torch
import torch.nn as nn
from typing import Literal


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_feedforward, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, padding_mask):
        return self.encoder(x, src_key_padding_mask=padding_mask)