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
        
    def _prep_embeddings_and_mask(self, signal: torch.Tensor, length: torch.Tensor):
        """
        Returns:
          embedded: [B, T+1, D]  (with CLS prepended, PE added, pads zeroed)
          padding_mask: [B, T+1] (True = pad)
          seq_len: int (original T without CLS)
        """
        batch_size, seq_len, _ = signal.shape
        seq_len_plus_cls = seq_len + 1

        # Input projection
        embedded = self.input_projection(signal)
        embedded = self.input_layernorm(embedded)

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        embedded = torch.cat((cls_token, embedded), dim=1)  # [B, T+1, D]

        # Build padding mask
        lengths_with_cls = length + 1
        padding_mask = torch.arange(seq_len_plus_cls, device=signal.device).expand(batch_size, -1) >= lengths_with_cls.unsqueeze(1)

        # Add positional encodings (excluding CLS)
        if self.encoding_type == "relative":
            rel_bias = self.positional_encoding(seq_len)       # [T, T, D]
            token_bias = rel_bias.sum(1)                       # [T, D]
            token_bias = token_bias.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, D]
            embedded[:, 1:] += token_bias * (~padding_mask[:, 1:]).unsqueeze(-1)
        else:
            abs_pe = self.positional_encoding(seq_len).to(embedded.device)  # [1, T, D]
            embedded[:, 1:] += abs_pe * (~padding_mask[:, 1:]).unsqueeze(-1)

        # Wipe padded slots completely
        embedded[padding_mask] = 0.0
        
        return embedded, padding_mask
    
    def forward(self, signal: torch.Tensor, length: torch.Tensor):
        """
        signal: Tensor of shape [batch_size, seq_len, input_dim]
        lengths: Tensor of shape [batch_size], unpadded lengths
        """
        embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)
    
        # Transformer encoder
        encoded = self.encoder(embedded, padding_mask)  # [B, T+1, D]

        # Outputs
        cls_representation = encoded[:, 0]  # [B, D]
        token_representation = encoded[:, 1:]  # [B, T, D]

        seq_logits = self.classifier(cls_representation)
        token_logits = self.token_classifier(token_representation)

        return {
            "classification": seq_logits,
            "segmentation": token_logits
        }

    @torch.no_grad()
    def get_cls_attention(self, signal: torch.Tensor, length: torch.Tensor, average_heads: bool = True):
        """
        Runs ONLY the encoder and returns last-layer CLS->token attention as [B, T].
        - Does not compute classification/segmentation heads.
        - Does not modify module state (uses a temporary forward hook).
        - average_heads=True returns the head-averaged map; if False and your
          PyTorch returns per-head weights, this will return [B, H, T].

        Returns:
          cls_to_tokens: [B, T]  (or [B, H, T] if average_heads=False and available)
        """
        self.eval()  # ensure deterministic layers like dropout are off

        embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)

        attn_cache = {"w": None}

        # get last encoder layer's MHA
        last_layer = self.encoder.encoder.layers[-1]
        mha = last_layer.self_attn

        # temporary hook to capture attention weights
        def _mha_hook(module, inputs, outputs):
            # outputs = (attn_output, attn_weights)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                attn_cache["w"] = outputs[1]

        handle = mha.register_forward_hook(_mha_hook)
        try:
            _ = self.encoder(embedded, padding_mask)  # run backbone only
        finally:
            handle.remove()

        attn = attn_cache["w"]
        if attn is None:
            # Fallback: some PyTorch versions can elide weights if not requested.
            # In stock nn.TransformerEncoderLayer they’re returned; if not, raise.
            raise RuntimeError("Could not capture attention weights from last layer.")

        # Normalize to common shapes
        # Possible shapes:
        #   [B, L_q, L_k]              (head-averaged)
        #   [B, num_heads, L_q, L_k]   (per-head)
        if attn.dim() == 4:  # per-head
            # pick CLS row and drop CLS column
            cls_to_all = attn[:, :, 0, :]     # [B, H, T+1]
            cls_to_tokens = cls_to_all[:, :, 1:]  # [B, H, T]
            # mask pads on key side
            key_pad = padding_mask[:, 1:].unsqueeze(1)  # [B, 1, T]
            cls_to_tokens = cls_to_tokens.masked_fill(key_pad, 0.0)
            # renormalize over unpadded keys
            denom = cls_to_tokens.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            cls_to_tokens = cls_to_tokens / denom
            if average_heads:
                cls_to_tokens = cls_to_tokens.mean(dim=1)  # [B, T]
            return cls_to_tokens
        elif attn.dim() == 3:  # head-averaged
            cls_to_all = attn[:, 0, :]      # [B, T+1]
            cls_to_tokens = cls_to_all[:, 1:]  # [B, T]
            key_pad = padding_mask[:, 1:]      # [B, T]
            cls_to_tokens = cls_to_tokens.masked_fill(key_pad, 0.0)
            denom = cls_to_tokens.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            cls_to_tokens = cls_to_tokens / denom
            return cls_to_tokens
        else:
            raise RuntimeError(f"Unexpected attention weight shape: {attn.shape}")
