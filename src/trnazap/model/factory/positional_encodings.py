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

    Caches cos/sin tables as non-trainable buffers and grows them
    dynamically if the sequence length exceeds the cache.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
               (Su et al., 2021)

    Args:
        head_dim:    Dimension per attention head (must be even).
        max_seq_len: Initial cache size — grows automatically if exceeded.
        base:        Frequency base (default 10000).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE."
        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self._cache_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)        # [T, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)      # [T, head_dim]
        # [1, 1, T, head_dim] — broadcasts over B and H
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            [B, H, T, head_dim]
            position_ids: [B, T] optional; if None, positions are 0..T-1

        Returns:
            [B, H, T, head_dim] with rotary embedding applied
        """
        B, H, T, D = x.shape

        # Grow cache if needed
        if T > self._cache_len:
            self._build_cache(T * 2)

        if position_ids is None:
            cos = self.cos_cache[:, :, :T, :]        # [1, 1, T, D]
            sin = self.sin_cache[:, :, :T, :]
        else:
            cos = self.cos_cache[0, 0][position_ids].unsqueeze(1)  # [B, 1, T, D]
            sin = self.sin_cache[0, 0][position_ids].unsqueeze(1)

        # Cast to input dtype — avoids implicit AMP casts allocating temp tensors
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # Fused rotation — split once, no separate _rotate_half allocation
        x1 = x[..., : D // 2]                       # [B, H, T, D//2]
        x2 = x[..., D // 2 :]                       # [B, H, T, D//2]

        return torch.cat([
            x1 * cos[..., : D // 2] - x2 * sin[..., : D // 2],
            x2 * cos[..., D // 2 :] + x1 * sin[..., D // 2 :],
        ], dim=-1)                                   # [B, H, T, D]