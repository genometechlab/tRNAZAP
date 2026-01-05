import torch
import torch.nn as nn
from typing import Literal
from typing import Optional


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
    
#----------------------------------------------
# The following classes are reserved for tRNAZAPformer
#----------------------------------------------

class EncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer (PyTorch-like naming):
      x -> LN -> MHA -> dropout1 -> residual
        -> LN -> FFN(linear1, activation, dropout, linear2) -> dropout2 -> residual
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(
        self,
        x: torch.Tensor,                                 # [B, L, D]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, L], True=pad
        attn_mask: Optional[torch.Tensor] = None,         # [L, L] or [B*nhead, L, L]
        need_weights: bool = False,
    ):
        # ---- Self-attention block (pre-norm) ----
        x_norm = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False if need_weights else True,
        )
        x = x + self.dropout1(attn_out)

        # ---- FFN block (pre-norm) ----
        x_norm = self.norm2(x)
        ffn = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ffn)

        return (x, attn_w) if need_weights else x


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,                               # [B, L, D]
        key_padding_mask: torch.Tensor | None = None,   # [B, L], True=pad
        attn_mask: torch.Tensor | None = None,          # [L, L] optional
        return_attn: bool = False,
    ):
        attn_all = [] if return_attn else None

        for layer in self.layers:
            if return_attn:
                x, attn = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=True,
                )
                attn_all.append(attn)  # each: [B, H, L, L]
            else:
                x = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                    need_weights=False,
                )
        return (x, attn_all) if return_attn else x