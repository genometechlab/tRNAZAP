import torch
import torch.nn as nn
import math
from typing import Optional


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
    
    
class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Computes sin/cos rotation matrices for a given sequence length and head dim.
    Cached as a non-trainable buffer — recomputed only when T grows beyond cache.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
               (Su et al., 2021)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        self.head_dim = head_dim
        self.base = base

        # Precompute frequencies: [head_dim // 2]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self._cache_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # [T, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [T, head_dim]
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)  # [1,1,T,D]
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)  # [1,1,T,D]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x:            [B, H, T, head_dim]
            position_ids: [B, T] optional; if None, positions are 0..T-1

        Returns:
            x with rotary embedding applied: [B, H, T, head_dim]
        """
        B, H, T, D = x.shape

        if T > self._cache_len:
            self._build_cache(T * 2)  # grow cache with headroom

        if position_ids is None:
            cos = self.cos_cache[:, :, :T, :]   # [1, 1, T, D]
            sin = self.sin_cache[:, :, :T, :]
        else:
            # position_ids: [B, T] → index into cache
            cos = self.cos_cache[0, 0][position_ids].unsqueeze(1)  # [B, 1, T, D]
            sin = self.sin_cache[0, 0][position_ids].unsqueeze(1)

        return x * cos + self._rotate_half(x) * sin