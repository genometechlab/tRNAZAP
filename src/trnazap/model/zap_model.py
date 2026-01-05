import torch
import torch.nn as nn
from typing import Optional, Literal, Iterable
from .factory import EncoderWrapper

Task = Literal["fragmentation", "classification", "segmentation"]

class tRNAZAPFormer(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        dropout_rate_transformer: float = 0.2,
        dropout_rate_fc: float = 0.2,
        max_seq_len: int = 1000,
        num_classification_classes: int = 22,
        num_segmentation_classes: int = 4,
        positional_encoding_type: Literal["learnable", "sinusoidal"] = "sinusoidal",
        enabled_tasks: Optional[Iterable[Task]] = None,  # NEW
    ):
        super().__init__()
        
        if enabled_tasks is None:
            enabled_tasks = ("fragmentation", "classification", "segmentation")
        self.enabled_tasks = set(enabled_tasks)

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
        else:
            raise ValueError("Invalid positional encoding type")

        self.encoding_type = positional_encoding_type

        # Transformer encoder
        self.encoder = EncoderWrapper(
            hidden_size, 
            num_heads, 
            dim_feedforward, 
            num_layers, 
            dropout_rate_transformer
        )

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
        lengths_with_cls = (length + 1).clamp_max(seq_len_plus_cls)
        padding_mask = torch.arange(seq_len_plus_cls, device=signal.device).expand(batch_size, -1) >= lengths_with_cls.unsqueeze(1)

        # Add positional encodings (excluding CLS)
        abs_pe = self.positional_encoding(seq_len).to(embedded.device)  # [1, T, D]
        embedded[:, 1:] += abs_pe * (~padding_mask[:, 1:]).unsqueeze(-1)

        # Wipe padded slots completely
        embedded[padding_mask] = 0.0
        
        return embedded, padding_mask

    def forward(self, 
                signal: torch.Tensor,
                length: torch.Tensor,
                tasks: Optional[Iterable[Task]] = None):
        """
        signal: Tensor of shape [batch_size, seq_len, input_dim]
        lengths: Tensor of shape [batch_size], unpadded lengths
        tasks: tasks at runtime
        """
        # Which tasks to run this call
        if tasks is None:
            tasks = self.enabled_tasks
        tasks = set(tasks)

        missing = tasks - self.enabled_tasks
        if missing:
            raise ValueError(f"Requested tasks not enabled in this model: {sorted(missing)}")
        
        embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)
    
        # Transformer encoder
        encoded = self.encoder(embedded, key_padding_mask=padding_mask)  # [B, T+1, D]


        out = {}
        if "classification" in tasks:
            cls_rep = encoded[:, 0]  # [B, D]
            out["classification"] = self.classifier(cls_rep)

        if "segmentation" in tasks:
            tok_rep = encoded[:, 1:]  # [B, T, D]
            out["segmentation"] = self.token_classifier(tok_rep)
            
        if "fragmentation" in tasks:
            cls_rep = encoded[:, 0]  # [B, D]
            tok_rep = encoded[:, 1:]  # [B, T, D]
            valid = (~padding_mask[:, 1:]).unsqueeze(-1)         # [B, T, 1]
            summed = (tok_rep * valid).sum(dim=1)                # [B, D]
            denom = valid.sum(dim=1).clamp_min(1.0)              # [B, 1]
            mean_pooled = summed / denom                         # [B, D]
            frag_rep = torch.cat([cls_rep, mean_pooled], dim=1)  # [B, 2D]
            out["fragmentation"] = self.frag_classifier(frag_rep)

        return out
    
    @torch.no_grad()
    def get_cls_attention(self, signal: torch.Tensor, length: torch.Tensor, average_heads: bool = True):
        """
        Uses attention weights from the LAST encoder layer.

        Returns:
        cls_attn:      [B, H, T] attention probabilities from CLS -> tokens (pads = 0)
        cls_attn_mean: [B, T]    head-averaged attention (if average_heads)
        """
        self.eval()
        embedded, padding_mask = self._prep_embeddings_and_mask(signal, length)  # [B, L, D], [B, L]
        encoded, attn_all = self.encoder(embedded, key_padding_mask=padding_mask, return_attn=True)

        last_attn = attn_all[-1]  # expected [B, H, L, L]
        if last_attn.dim() == 3:
            # If encoder returns averaged weights [B, L, L], expand to [B, 1, L, L]
            last_attn = last_attn.unsqueeze(1)

        # CLS is position 0
        cls_attn = last_attn[:, :, 0, 1:]  # [B, H, T]

        token_pad = padding_mask[:, 1:]  # [B, T], True=pad
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
        reduce: str = "l2",          # "l2" or "sum"
        normalize: bool = True,
    ):
        """
        Gradient-based saliency for the classification output w.r.t. the *input signal*.

        Args:
          signal: [B, T, Din]
          length: [B]
          target_class:
            - None: uses argmax class per example
            - int: uses the same class for all examples
            - Tensor [B]: per-example class indices
          use_abs: take abs of gradients before reduction
          reduce: reduce gradient across feature dim -> token score ("l2" or "sum")
          normalize: normalize saliency per example to sum to 1 over non-pad tokens

        Returns:
          saliency: [B, T] (pads = 0)
          chosen_class: [B] target class indices used
        """
        if self.classifier is None:
            raise RuntimeError("get_token_saliency requires 'classification' to be enabled in this model instance.")

        self.eval()

        # Ensure we can take gradients w.r.t. the input
        x = signal.detach().clone()
        x.requires_grad_(True)

        out = self.forward(x, length, tasks=("classification",))
        logits = out["classification"]  # [B, C]
        B, T, _ = x.shape

        # Determine target class indices
        if target_class is None:
            chosen = logits.argmax(dim=-1)  # [B]
        elif isinstance(target_class, int):
            chosen = torch.full((B,), target_class, device=logits.device, dtype=torch.long)
        else:
            chosen = target_class.to(logits.device).long()
            if chosen.ndim != 1 or chosen.shape[0] != B:
                raise ValueError("target_class tensor must have shape [B]")

        selected = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [B]
        loss = selected.sum()

        self.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        g = x.grad
        if g is None:
            raise RuntimeError("No gradients computed. Ensure signal requires_grad and forward uses it.")

        if use_abs:
            g = g.abs()

        if reduce == "l2":
            sal = torch.sqrt((g ** 2).sum(dim=-1) + 1e-12)
        elif reduce == "sum":
            sal = g.sum(dim=-1)
        else:
            raise ValueError("reduce must be 'l2' or 'sum'")

        # Mask pads
        pad_mask = torch.arange(T, device=signal.device).unsqueeze(0) >= length.unsqueeze(1)  # [B, T]
        sal = sal.masked_fill(pad_mask, 0.0)

        if normalize:
            denom = sal.sum(dim=1, keepdim=True).clamp_min(1e-12)
            sal = sal / denom

        return sal.detach(), chosen.detach()