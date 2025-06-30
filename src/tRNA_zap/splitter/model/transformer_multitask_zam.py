import torch
import torch.nn as nn
from typing import Literal

from .factory import TransformerEncoderWrapper

class TransformerZAM_multitask(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        num_classes: int = 22,
        num_classes_seq2seq: int = 4,
        dropout_rate_transformer: float = 0.2,
        dropout_rate_fc: float = 0.2,
        max_seq_len: int = 1000,
        positional_encoding_type: Literal["learnable", "sinusoidal", "relative"] = "sinusoidal",
    ):
        super().__init__()

        # Input projection and normalization
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Positional encoding module
        if positional_encoding_type == "learnable":
            from .factory import LearnablePositionalEncoding
            self.positional_encoding = LearnablePositionalEncoding(max_seq_len, hidden_size)
        elif positional_encoding_type == "sinusoidal":
            from .factory import SinusoidalPositionalEncoding
            self.positional_encoding = SinusoidalPositionalEncoding(max_seq_len, hidden_size)
        elif positional_encoding_type == "relative":
            from .factory import RelativePositionalEncoding
            self.positional_encoding = RelativePositionalEncoding(max_seq_len, hidden_size)
        else:
            raise ValueError("Invalid positional encoding type")

        self.encoding_type = positional_encoding_type

        # Transformer encoder
        self.encoder = TransformerEncoderWrapper(
            hidden_size, num_heads, dim_feedforward, num_layers, dropout_rate_transformer
        )

        # Sequence classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(128, num_classes),
        )

        # Token-level classification head
        self.token_classifier = nn.Linear(hidden_size, num_classes_seq2seq)

    def forward(self, signal: torch.Tensor, length: torch.Tensor):
        """
        signal: Tensor of shape [batch_size, seq_len, input_dim]
        lengths: Tensor of shape [batch_size], unpadded lengths
        """
        batch_size, seq_len, _ = signal.shape
        seq_len_plus_cls = seq_len + 1

        # Input projection
        embedded = self.input_projection(signal)
        embedded = self.input_layernorm(embedded)

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embedded = torch.cat((cls_token, embedded), dim=1)  # [B, T+1, D]

        # Add positional encodings (excluding CLS)
        if self.encoding_type == "relative":
            rel_bias = self.positional_encoding(seq_len)  # [T, T, D]
            token_part = embedded[:, 1:] + rel_bias.sum(dim=1).unsqueeze(0).to(embedded.device)
            embedded = torch.cat((embedded[:, :1], token_part), dim=1)
        else:
            abs_pe = self.positional_encoding(seq_len).to(embedded.device)  # [1, T, D]
            embedded[:, 1:] = embedded[:, 1:] + abs_pe  # no PE for CLS

        # Build padding mask
        lengths_with_cls = length + 1
        padding_mask = torch.arange(seq_len_plus_cls, device=signal.device).expand(batch_size, -1) >= lengths_with_cls.unsqueeze(1)

        # Transformer encoder
        encoded = self.encoder(embedded, padding_mask)  # [B, T+1, D]

        # Outputs
        cls_representation = encoded[:, 0]  # [B, D]
        token_representation = encoded[:, 1:]  # [B, T, D]

        seq_logits = self.classifier(cls_representation)
        token_logits = self.token_classifier(token_representation)

        return {
            "seq_class": seq_logits,
            "seq2seq": token_logits
        }
