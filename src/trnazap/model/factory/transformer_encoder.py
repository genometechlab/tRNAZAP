import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional, Tuple
from .positional_encodings import RoPEEmbedding


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

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention with optional RoPE support.

    Args:
        embed_dim:    Total embedding dimension.
        num_heads:    Number of attention heads.
        dropout:      Attention dropout probability.
        batch_first:  If True, input/output tensors are [B, T, D].
                      Must be True to match the rest of the codebase.
        use_rope:     If True, apply RoPE to Q and K before attention.
        rope_base:    Base frequency for RoPE (default 10000).
        max_seq_len:  Initial RoPE cache size (grows dynamically if needed).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_rope: bool = False,
        rope_base: int = 10000,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert batch_first, "MultiheadAttention only supports batch_first=True."
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope

        # Packed QKV projection — same layout as nn.MultiheadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))

        # Output projection — same name as nn.MultiheadAttention for compat
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RoPEEmbedding(
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
                base=rope_base,
            )
        else:
            self.rope = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier uniform init — matches nn.MultiheadAttention default."""
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _project_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project inputs to Q, K, V using packed in_proj_weight.
        For self-attention query == key == value.
        """
        W_q, W_k, W_v = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        Q = F.linear(query, W_q, b_q)
        K = F.linear(key,   W_k, b_k)
        V = F.linear(value, W_v, b_v)
        return Q, K, V

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, H, T, head_dim]"""
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, head_dim] → [B, T, D]"""
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,                          # [B, T, D]
        key: torch.Tensor,                            # [B, S, D]
        value: torch.Tensor,                          # [B, S, D]
        key_padding_mask: Optional[torch.Tensor] = None,   # [B, S] True=pad
        attn_mask: Optional[torch.Tensor] = None,          # [T, S] or [B*H, T, S]
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query:              [B, T, D]
            key:                [B, S, D]
            value:              [B, S, D]
            key_padding_mask:   [B, S]      True = ignore (pad)
            attn_mask:          [T, S]      additive mask (−inf = block)
            need_weights:       return attention weights
            average_attn_weights: average weights over heads if need_weights

        Returns:
            output:   [B, T, D]
            attn_w:   [B, H, T, S] or [B, T, S] or None
        """
        B, T, _ = query.shape
        S = key.shape[1]
        scale = math.sqrt(self.head_dim)

        # Project and split heads
        Q, K, V = self._project_qkv(query, key, value)
        Q = self._split_heads(Q)   # [B, H, T, head_dim]
        K = self._split_heads(K)   # [B, H, S, head_dim]
        V = self._split_heads(V)   # [B, H, S, head_dim]

        # Apply RoPE to Q and K
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        if need_weights:
            # --- Explicit path: materialize scores to return attention weights ---
            # Used by get_cls_attention() and any interpretability call.
            # Memory intensive but only triggered on demand, never during training
            # or standard inference.

            # Build combined additive mask
            combined_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    combined_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,S]
                elif attn_mask.dim() == 3:
                    combined_mask = attn_mask.unsqueeze(1)               # [B,1,T,S]

            if key_padding_mask is not None:
                pad_mask = key_padding_mask[:, None, None, :].expand(B, self.num_heads, T, S)
                inf_mask = torch.zeros_like(pad_mask, dtype=Q.dtype).masked_fill(pad_mask, float("-inf"))
                combined_mask = inf_mask if combined_mask is None else combined_mask + inf_mask

            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # [B, H, T, S]
            if combined_mask is not None:
                scores = scores + combined_mask

            attn_w = torch.softmax(scores, dim=-1)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)              # guard full-pad rows
            attn_w = self.attn_dropout(attn_w)

            out = torch.matmul(attn_w, V)                           # [B, H, T, head_dim]
            out = self._merge_heads(out)
            out = self.out_proj(out)

            return out, attn_w.mean(dim=1) if average_attn_weights else attn_w

        else:
            # --- Fast path: fused FlashAttention via F.scaled_dot_product_attention ---
            # Never materializes the full [B, H, T, S] score matrix.
            # Handles masking internally.

            # Build combined boolean/float mask for SDPA
            # SDPA expects an additive float mask [B, H, T, S] or [1, 1, T, S]
            combined_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    combined_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    combined_mask = attn_mask.unsqueeze(1)

            if key_padding_mask is not None:
                pad_mask = key_padding_mask[:, None, None, :].expand(B, self.num_heads, T, S)
                inf_mask = torch.zeros_like(pad_mask, dtype=Q.dtype).masked_fill(pad_mask, float("-inf"))
                combined_mask = inf_mask if combined_mask is None else combined_mask + inf_mask

            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=combined_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )                                                        # [B, H, T, head_dim]
            out = self._merge_heads(out)
            out = self.out_proj(out)

            return out, None

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
        norm_first: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first

        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
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
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        if self.norm_first:
            x_norm = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=not need_weights,
            )
            x = x + self.dropout1(attn_out)
            x_norm = self.norm2(x)
            ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            x = x + self.dropout2(ffn_out)
        else:
            attn_out, attn_w = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=not need_weights,
            )
            x = x + self.dropout1(attn_out)
            x = self.norm1(x)
            ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout2(ffn_out)
            x = self.norm2(x)

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
        norm_first: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 4096,
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
                norm_first=norm_first,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,                              # [B, L, D]
        key_padding_mask: torch.Tensor | None = None, # [B, L], True=pad
        attn_mask: torch.Tensor | None = None,        # [L, L] optional
        need_weights: bool = False,
    ):
        attn_all = [] if need_weights else None

        for layer in self.layers:
            if need_weights:
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

        return (x, attn_all) if need_weights else x