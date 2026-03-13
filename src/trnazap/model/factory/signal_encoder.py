import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Literal, Optional


class SignalEncoder(ABC, nn.Module):
    """
    Abstract base class for signal feature extractors.
    All subclasses consume a raw 1-D signal [B, N] and produce
    a sequence of token embeddings [B, T, out_channels].
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels  # C_out fed into input_projection

    @abstractmethod
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: [B, N]  raw signal (already z-scored, float32/16)
        Returns:
            tokens: [B, T, out_channels]
        """

    @property
    @abstractmethod
    def effective_stride(self) -> int:
        """
        Number of raw signal samples consumed per output token.
        Used externally to validate config consistency.
        """


# ---------------------------------------------------------------------------

class IdentitySignalEncoder(SignalEncoder):
    """
    Equivalent to the reshaping signal
      [B, N] → [B, T, chunk_size]   (non-overlapping windows)
    No learnable parameters; out_channels == chunk_size.

    The single Conv1d with kernel_size=stride=chunk_size is mathematically
    identical to slicing non-overlapping windows and projecting with a
    weight-tied linear layer — but here we skip the projection and let
    the model's existing input_projection handle it.
    """

    def __init__(self, chunk_size: int):
        super().__init__(out_channels=chunk_size)
        self.chunk_size = chunk_size
        # Non-overlapping 1-D convolution — identity feature extraction
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=chunk_size,
            kernel_size=chunk_size,
            stride=chunk_size,
            bias=False,
        )
        # Fix weights to extract raw windows (identity mapping)
        self._init_identity_weights()

    @torch.no_grad()
    def _init_identity_weights(self):
        """
        Each output channel i picks up exactly sample i within the window.
        W[i, 0, i] = 1.0, all others = 0.
        """
        self.conv.weight.zero_()
        for i in range(self.chunk_size):
            self.conv.weight[i, 0, i] = 1.0
        self.conv.weight.requires_grad_(False)

    @property
    def effective_stride(self) -> int:
        return self.chunk_size

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: [B, N]
        x = signal.unsqueeze(1)                    # [B, 1, N]
        x = self.conv(x)                           # [B, chunk_size, T]
        return x.permute(0, 2, 1).contiguous()     # [B, T, chunk_size]


# ---------------------------------------------------------------------------

class ConvSignalEncoder(SignalEncoder):
    """
    Learnable multi-layer Conv1d feature extractor.
    Inspired by Wav2Vec 2.0 / HuBERT frontend design.

    Each layer applies:
        Conv1d → LayerNorm (over channels) → Activation

    Args:
        channels:     List of channel sizes including input.
                      len(channels) - 1 == number of conv layers.
                      channels[0] must be 1 (mono signal).
                      e.g. [1, 32, 64, 128]
        kernel_sizes: Conv kernel size per layer. len == len(channels) - 1.
        strides:      Conv stride per layer.      len == len(channels) - 1.
        activation:   "relu" or "gelu"
    """

    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        activation: Literal["relu", "gelu"] = "gelu",
    ):
        assert channels[0] == 1, "First channel must be 1 (raw signal)."
        assert len(channels) - 1 == len(kernel_sizes) == len(strides), (
            "channels, kernel_sizes, and strides are inconsistent. "
            "Expected len(channels) - 1 == len(kernel_sizes) == len(strides)."
        )

        super().__init__(out_channels=channels[-1])

        act_fn = nn.GELU if activation == "gelu" else nn.ReLU

        layers = []
        for i, (c_in, c_out, k, s) in enumerate(
            zip(channels[:-1], channels[1:], kernel_sizes, strides)
        ):
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, bias=False, padding=k-s))
            layers.append(_TransposedLayerNorm(c_out))  # LN over channel dim
            layers.append(act_fn())

        self.encoder = nn.Sequential(*layers)
        self._strides = strides
        self._kernel_sizes = kernel_sizes

    @property
    def effective_stride(self) -> int:
        """Product of all strides = samples consumed per output token."""
        stride = 1
        for s in self._strides:
            stride *= s
        return stride

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: [B, N]
        x = signal.unsqueeze(1)             # [B, 1, N]
        x = self.encoder(x)                 # [B, C_out, T]
        return x.permute(0, 2, 1).contiguous()  # [B, T, C_out]


# ---------------------------------------------------------------------------

class _TransposedLayerNorm(nn.Module):
    """
    LayerNorm applied over the channel dimension of a [B, C, T] tensor.
    Transposes to [B, T, C], normalizes, then transposes back.
    Keeps the conv stack interface clean.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------

def build_signal_encoder(
    stem_type: Literal["identity", "conv"],
    chunk_size: Optional[int] = None,
    stem_channels: Optional[List[int]] = None,
    stem_kernel_sizes: Optional[List[int]] = None,
    stem_strides: Optional[List[int]] = None,
    stem_activation: Literal["relu", "gelu"] = "gelu",
    effective_stride: Optional[int] = None,
) -> SignalEncoder:
    """
    Factory function — mirrors the positional encoding factory pattern.

    Args:
        stem_type:         "identity" or "conv"
        chunk_size:        Required for "identity"
        stem_channels:     Required for "conv", e.g. [1, 32, 64, 128]
        stem_kernel_sizes: Required for "conv"
        stem_strides:      Required for "conv"
        stem_activation:   "relu" or "gelu" (conv only)
        effective_stride:  Optional; if provided, validates against
                           product of stem_strides for "conv" stem.

    Returns:
        SignalEncoder instance
    """
    if stem_type == "identity":
        if chunk_size is None:
            raise ValueError("chunk_size is required for IdentitySignalEncoder.")
        return IdentitySignalEncoder(chunk_size=chunk_size)

    elif stem_type == "conv":
        if any(v is None for v in [stem_channels, stem_kernel_sizes, stem_strides]):
            raise ValueError(
                "stem_channels, stem_kernel_sizes, and stem_strides "
                "are all required for ConvSignalEncoder."
            )
        encoder = ConvSignalEncoder(
            channels=stem_channels,
            kernel_sizes=stem_kernel_sizes,
            strides=stem_strides,
            activation=stem_activation,
        )
        if effective_stride is not None and encoder.effective_stride != effective_stride:
            raise ValueError(
                f"Product of stem_strides ({encoder.effective_stride}) does not match "
                f"effective_stride ({effective_stride}) specified in config. "
                f"stem_strides={stem_strides}"
            )
        return encoder

    else:
        raise ValueError(f"Unknown stem_type '{stem_type}'. Expected 'identity' or 'conv'.")