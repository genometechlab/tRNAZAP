import torch
import torch.nn as nn
from typing import Optional, Literal, Iterable, List
from .factory import EncoderWrapper
from .factory import SignalEncoder, IdentitySignalEncoder, build_signal_encoder

Task = Literal["fragmentation", "classification", "segmentation"]


class tRNAZAPFormer(nn.Module):
    def __init__(
        self,
        # --- Signal encoder ---
        stem_type: Literal["identity", "conv"] = "identity",
        chunk_size: int = 64,
        stem_channels: Optional[List[int]] = None,
        stem_kernel_sizes: Optional[List[int]] = None,
        stem_strides: Optional[List[int]] = None,
        stem_activation: Literal["relu", "gelu"] = "gelu",
        effective_stride: Optional[int] = None,
        # --- Transformer ---
        hidden_size: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        dropout_rate_transformer: float = 0.2,
        dropout_rate_fc: float = 0.2,
        max_seq_len: int = 1000,
        # --- Task heads ---
        num_classification_classes: int = 22,
        num_segmentation_classes: int = 4,
        # --- Positional encoding ---
        positional_encoding_type: Literal["learnable", "sinusoidal", "rope"] = "sinusoidal",
        # --- Tasks ---
        enabled_tasks: Optional[Iterable[Task]] = None,
    ):
        super().__init__()

        if enabled_tasks is None:
            enabled_tasks = ("fragmentation", "classification", "segmentation")
        self.enabled_tasks = set(enabled_tasks)
        use_rope = positional_encoding_type == "rope"
        self.use_rope = use_rope

        # ------------------------------------------------------------------
        # Signal encoder
        # ------------------------------------------------------------------
        self.signal_encoder: SignalEncoder = build_signal_encoder(
            stem_type=stem_type,
            chunk_size=chunk_size,
            stem_channels=stem_channels,
            stem_kernel_sizes=stem_kernel_sizes,
            stem_strides=stem_strides,
            stem_activation=stem_activation,
            effective_stride=effective_stride,
        )
        encoder_out_channels = self.signal_encoder.out_channels

        # ------------------------------------------------------------------
        # Input projection and normalization
        # ------------------------------------------------------------------
        self.input_projection = nn.Linear(encoder_out_channels, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # ------------------------------------------------------------------
        # Positional encoding — skipped when RoPE is active
        # ------------------------------------------------------------------
        self.positional_encoding = None
        if not use_rope:
            if positional_encoding_type == "learnable":
                from .factory import LearnablePositionalEncoding
                self.positional_encoding = LearnablePositionalEncoding(max_seq_len, hidden_size)
            elif positional_encoding_type == "sinusoidal":
                from .factory import SinusoidalPositionalEncoding
                self.positional_encoding = SinusoidalPositionalEncoding(max_seq_len, hidden_size)
            else:
                raise ValueError(
                    f"Invalid positional_encoding_type '{positional_encoding_type}'. "
                    "Expected 'sinusoidal', 'learnable', or 'rope'."
                )

        self.encoding_type = positional_encoding_type

        # ------------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------------
        self.encoder = EncoderWrapper(
            hidden_size,
            num_heads,
            dim_feedforward,
            num_layers,
            dropout_rate_transformer,
            norm_first=False,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        # ------------------------------------------------------------------
        # Task heads
        # ------------------------------------------------------------------
        self.frag_classifier = None
        if "fragmentation" in self.enabled_tasks:
            self.frag_classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate_fc),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate_fc),
                nn.Linear(128, 2),
            )

        self.classifier = None
        if "classification" in self.enabled_tasks:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate_fc),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate_fc),
                nn.Linear(128, num_classification_classes),
            )

        self.token_classifier = None
        if "segmentation" in self.enabled_tasks:
            self.token_classifier = nn.Linear(hidden_size, num_segmentation_classes)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """[B, N] → [B, T, encoder_out_channels]"""
        return self.signal_encoder(signal)

    def _raw_length_to_tokens(self, raw_length: torch.Tensor) -> torch.Tensor:
        """Raw sample count → token count."""
        return (raw_length // self.signal_encoder.effective_stride).clamp_min(0)

    def _prep_embeddings_and_mask(
        self,
        signal: torch.Tensor,
        length: torch.Tensor,
    ):
        """
        Args:
            signal: [B, N] raw signal
            length: [B]    number of valid raw samples

        Returns:
            embedded:     [B, T+1, hidden_size]
            padding_mask: [B, T+1]  True = pad
            token_length: [B]
        """
        tokens = self._encode_signal(signal)             # [B, T, C]
        token_length = self._raw_length_to_tokens(length)

        batch_size, seq_len, _ = tokens.shape
        seq_len_plus_cls = seq_len + 1

        # Project + normalize
        embedded = self.input_projection(tokens)         # [B, T, D]
        embedded = self.input_layernorm(embedded)

        # Prepend CLS token
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embedded = torch.cat((cls_token, embedded), dim=1)  # [B, T+1, D]

        # Padding mask
        lengths_with_cls = (token_length + 1).clamp_max(seq_len_plus_cls)
        padding_mask = (
            torch.arange(seq_len_plus_cls, device=signal.device)
            .expand(batch_size, -1) >= lengths_with_cls.unsqueeze(1)
        )

        # Absolute positional encodings — skipped when RoPE is active
        if self.positional_encoding is not None:
            abs_pe = self.positional_encoding(seq_len).to(embedded.device)  # [1, T, D]
            embedded[:, 1:] += abs_pe * (~padding_mask[:, 1:]).unsqueeze(-1)

        # Zero out padded slots
        embedded[padding_mask] = 0.0

        return embedded, padding_mask, token_length

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        signal: torch.Tensor,
        length: torch.Tensor,
        tasks: Optional[Iterable[Task]] = None,
    ):
        """
        Args:
            signal: [B, N] raw signal
            length: [B]    number of valid raw samples
            tasks:  subset of enabled_tasks; None = all enabled
        """
        if tasks is None:
            tasks = self.enabled_tasks
        tasks = set(tasks)

        missing = tasks - self.enabled_tasks
        if missing:
            raise ValueError(
                f"Requested tasks not enabled in this model: {sorted(missing)}"
            )

        embedded, padding_mask, _ = self._prep_embeddings_and_mask(signal, length)
        encoded = self.encoder(embedded, key_padding_mask=padding_mask)  # [B, T+1, D]

        out = {}

        if "classification" in tasks:
            cls_rep = encoded[:, 0]
            out["classification"] = self.classifier(cls_rep)

        if "segmentation" in tasks:
            tok_rep = encoded[:, 1:]
            out["segmentation"] = self.token_classifier(tok_rep)

        if "fragmentation" in tasks:
            cls_rep = encoded[:, 0]
            tok_rep = encoded[:, 1:]
            valid = (~padding_mask[:, 1:]).unsqueeze(-1)
            summed = (tok_rep * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            mean_pooled = summed / denom
            frag_rep = torch.cat([cls_rep, mean_pooled], dim=1)
            out["fragmentation"] = self.frag_classifier(frag_rep)

        return out

    # ----------------------------------------------------------------------
    # Interpretability
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def get_cls_attention(
        self,
        signal: torch.Tensor,
        length: torch.Tensor,
        average_heads: bool = True,
    ):
        self.eval()
        embedded, padding_mask, _ = self._prep_embeddings_and_mask(signal, length)
        encoded, attn_all = self.encoder(
            embedded, key_padding_mask=padding_mask, need_weights=True
        )

        last_attn = attn_all[-1]
        if last_attn.dim() == 3:
            last_attn = last_attn.unsqueeze(1)

        cls_attn = last_attn[:, :, 0, 1:]          # [B, H, T]
        token_pad = padding_mask[:, 1:]
        if token_pad.any():
            cls_attn = cls_attn.masked_fill(token_pad.unsqueeze(1), 0.0)

        cls_attn_mean = cls_attn.mean(dim=1) if average_heads else None
        return cls_attn, cls_attn_mean

    def get_token_saliency(
        self,
        signal: torch.Tensor,
        length: torch.Tensor,
        target_class: torch.Tensor | int | None = None,
        use_abs: bool = True,
        reduce: str = "l2",
        normalize: bool = True,
    ):
        if self.classifier is None:
            raise RuntimeError(
                "get_token_saliency requires 'classification' to be enabled."
            )

        self.eval()

        x = signal.detach().clone()
        x.requires_grad_(True)

        out = self.forward(x, length, tasks=("classification",))
        logits = out["classification"]
        B = x.shape[0]

        token_length = self._raw_length_to_tokens(length)
        T = self._encode_signal(x.detach()).shape[1]

        if target_class is None:
            chosen = logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            chosen = torch.full((B,), target_class, device=logits.device, dtype=torch.long)
        else:
            chosen = target_class.to(logits.device).long()
            if chosen.ndim != 1 or chosen.shape[0] != B:
                raise ValueError("target_class tensor must have shape [B]")

        selected = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
        loss = selected.sum()

        self.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        g = x.grad
        if g is None:
            raise RuntimeError(
                "No gradients computed. Ensure signal requires_grad and forward uses it."
            )

        if use_abs:
            g = g.abs()

        # Raw signal [B, N]: reshape gradients into [B, T, stride] then reduce
        stride = self.signal_encoder.effective_stride
        g = g[:, : T * stride].reshape(B, T, stride)

        if reduce == "l2":
            sal = torch.sqrt((g ** 2).sum(dim=-1) + 1e-12)
        elif reduce == "sum":
            sal = g.sum(dim=-1)
        else:
            raise ValueError("reduce must be 'l2' or 'sum'")

        pad_mask = (
            torch.arange(T, device=signal.device).unsqueeze(0) >= token_length.unsqueeze(1)
        )
        sal = sal.masked_fill(pad_mask, 0.0)

        if normalize:
            denom = sal.sum(dim=1, keepdim=True).clamp_min(1e-12)
            sal = sal / denom

        return sal.detach(), chosen.detach()