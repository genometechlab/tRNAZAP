import os
import glob
import math
import torch
from dataclasses import dataclass, fields, MISSING
from typing import Optional, Dict, Any, Union, List, Protocol
import yaml
import json
import warnings
import importlib

from pathlib import Path
from ..utils import load_weights


PathLike = Union[str, Path]


class ModelProtocol(Protocol):
    def forward(self, *args, **kwargs): ...


@dataclass
class ModelConfig:
    """Configuration class for tRNAZAPFormer model parameters."""

    # -----------------------------------------------------------------------
    # Model architecture — required
    # -----------------------------------------------------------------------
    chunk_size: int
    max_seq_len: int
    num_classification_classes: int
    num_segmentation_classes: int
    num_heads: int
    num_layers: int
    hidden_size: int
    dim_feedforward: int
    dropout: float

    # -----------------------------------------------------------------------
    # Signal encoder stem
    # -----------------------------------------------------------------------
    stem_type: str = "identity"                    # "identity" | "conv"
    stem_channels: Optional[List[int]] = None      # e.g. [1, 32, 64, 128]
    stem_kernel_sizes: Optional[List[int]] = None  # per-layer kernel sizes
    stem_strides: Optional[List[int]] = None       # per-layer strides
    stem_activation: str = "gelu"                  # "relu" | "gelu"

    # -----------------------------------------------------------------------
    # Positional encoding
    # -----------------------------------------------------------------------
    positional_encoding_type: str = "sinusoidal"   # "sinusoidal" | "learnable" | "rope"

    # -----------------------------------------------------------------------
    # Model information
    # -----------------------------------------------------------------------
    model_name: str = "tRNAZAPFormer"

    # -----------------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------------
    float_dtype: str = "float32"
    checkpoint_path: Optional[str] = None
    work_dir: Optional[str] = None
    label_names: Optional[dict] = None

    # -----------------------------------------------------------------------
    # Post-init validation
    # -----------------------------------------------------------------------
    def __post_init__(self):
        self._validate_stem()
        self._validate_positional_encoding()

    def _validate_stem(self) -> None:
        if self.stem_type == "identity":
            return

        if self.stem_type == "conv":
            missing = [
                name for name, val in [
                    ("stem_channels",     self.stem_channels),
                    ("stem_kernel_sizes", self.stem_kernel_sizes),
                    ("stem_strides",      self.stem_strides),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"stem_type='conv' requires: {', '.join(missing)}"
                )

            n_layers = len(self.stem_channels) - 1
            if len(self.stem_kernel_sizes) != n_layers:
                raise ValueError(
                    f"stem_kernel_sizes must have {n_layers} entries "
                    f"(len(stem_channels) - 1), got {len(self.stem_kernel_sizes)}."
                )
            if len(self.stem_strides) != n_layers:
                raise ValueError(
                    f"stem_strides must have {n_layers} entries "
                    f"(len(stem_channels) - 1), got {len(self.stem_strides)}."
                )
            if self.stem_channels[0] != 1:
                raise ValueError(
                    "stem_channels[0] must be 1 (single-channel raw signal)."
                )

            stride_product = math.prod(self.stem_strides)
            if stride_product != self.chunk_size:
                raise ValueError(
                    f"Product of stem_strides ({stride_product}) must equal "
                    f"chunk_size ({self.chunk_size}). "
                    f"stem_strides={self.stem_strides}"
                )
            return

        raise ValueError(
            f"Unknown stem_type '{self.stem_type}'. Expected 'identity' or 'conv'."
        )

    def _validate_positional_encoding(self) -> None:
        valid_pe = {"sinusoidal", "learnable", "rope"}
        if self.positional_encoding_type not in valid_pe:
            raise ValueError(
                f"Unknown positional_encoding_type '{self.positional_encoding_type}'. "
                f"Expected one of {valid_pe}."
            )

    # -----------------------------------------------------------------------
    # Derived property
    # -----------------------------------------------------------------------
    @property
    def effective_stride(self) -> int:
        """Samples per output token. Always equals chunk_size."""
        return self.chunk_size

    # -----------------------------------------------------------------------
    # Field validation helper
    # -----------------------------------------------------------------------
    @classmethod
    def _check_fields(cls, dict_: dict) -> dict:
        valid_config = {}
        invalid_keys = []
        valid_fields = {f.name for f in fields(cls)}

        for k, v in dict_.items():
            if k in valid_fields:
                valid_config[k] = v
            else:
                invalid_keys.append(k)

        required_fields = {
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required_fields - valid_config.keys()
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")

        if invalid_keys:
            warnings.warn(
                f"Model config received the following invalid keys: {', '.join(invalid_keys)}"
            )

        return valid_config

    @staticmethod
    def _resolve_paths(config: dict, base_dir: Path) -> dict:
        resolved = {}
        for k, v in config.items():
            if isinstance(v, str) and (k.endswith("_path") or k.endswith("_dir")):
                path = Path(v)
                if not path.is_absolute():
                    resolved[k] = str((base_dir / path).resolve())
                else:
                    resolved[k] = str(path)
            else:
                resolved[k] = v
        return resolved

    # -----------------------------------------------------------------------
    # Loaders
    # -----------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> "ModelConfig":
        yaml_path = Path(yaml_path).resolve()
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        base_dir = yaml_path.parent
        config_dict = cls._resolve_paths(config_dict, base_dir)
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: PathLike) -> "ModelConfig":
        json_path = Path(json_path).resolve()
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        base_dir = json_path.parent
        config_dict = cls._resolve_paths(config_dict, base_dir)
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)

    @classmethod
    def load_config(cls, cfg: Union["ModelConfig", str, Dict]) -> "ModelConfig":
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, str):
            cfg_path = Path(cfg)
            suffix = cfg_path.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                return cls.from_yaml(cfg)
            elif suffix == ".json":
                return cls.from_json(cfg)
            raise ValueError(f"Unsupported config file type: {suffix}")
        if isinstance(cfg, dict):
            return cls.from_dict(cfg)
        raise TypeError("config must be ModelConfig | str | dict")

    # -----------------------------------------------------------------------
    # Serializers
    # -----------------------------------------------------------------------
    def to_yaml(self, yaml_path: PathLike) -> None:
        yaml_path = Path(yaml_path)
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_json(self, json_path: PathLike) -> None:
        json_path = Path(json_path)
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# ---------------------------------------------------------------------------

class ModelLoader:

    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self) -> ModelProtocol:
        module = importlib.import_module("trnazap.model")
        model_class = getattr(module, self.config.model_name)

        model = model_class(
            # Signal encoder
            stem_type=self.config.stem_type,
            chunk_size=self.config.chunk_size,
            stem_channels=self.config.stem_channels,
            stem_kernel_sizes=self.config.stem_kernel_sizes,
            stem_strides=self.config.stem_strides,
            stem_activation=self.config.stem_activation,
            effective_stride=self.config.effective_stride,
            # Transformer
            max_seq_len=self.config.max_seq_len,
            num_classification_classes=self.config.num_classification_classes,
            num_segmentation_classes=self.config.num_segmentation_classes,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            dim_feedforward=self.config.dim_feedforward,
            # Positional encoding
            positional_encoding_type=self.config.positional_encoding_type,
        )

        if self.config.float_dtype == "float64":
            model = model.double()

        return model

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is None and self.config.work_dir:
            best_weights = glob.glob(
                os.path.join(self.config.work_dir, "best_epoch_*.pth")
            )
            if best_weights:
                checkpoint_path = best_weights[-1]
            else:
                raise ValueError(f"No checkpoint found in {self.config.work_dir}")

        if checkpoint_path is None:
            warnings.warn("No checkpoint path provided", stacklevel=2)
        else:
            load_weights(self.model, checkpoint_path)

    def get_model(self, load_checkpoint: bool = True) -> ModelProtocol:
        if self.model is None:
            self.model = self.build_model()
            self.model.to(self.device)

        if load_checkpoint:
            self.load_checkpoint()

        return self.model

    def get_num_parameters(self) -> int:
        if self.model is None:
            self.model = self.build_model()
        return sum(p.numel() for p in self.model.parameters())