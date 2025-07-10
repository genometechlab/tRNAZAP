import os
import glob
import torch
from dataclasses import dataclass, field, fields, MISSING
from typing import Optional, Dict, Any, Union
import yaml
import json
import warnings

from pathlib import Path
from ..model import TransformerZAM_multitask
from ..utils import load_weights


PathLike = Union[str, Path]

@dataclass
class ModelConfig:
    """Configuration class for TransformerZAM model parameters."""
    # Model architecture parameters
    chunk_size: int
    max_seq_len: int
    num_classes: int
    num_classes_seq2seq: int
    nhead: int
    num_layers: int
    hidden_size: int
    dim_feedforward: int
    positional_encoding_type: str
    dropout: float

    # Model information
    model_name: str

    # Training parameters
    float_dtype: str
    
    # Model checkpoint
    checkpoint_path: str
    work_dir: Optional[str] = None

    # Labels names
    label_names: Optional[dict] = None

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

        if len(invalid_keys)>0:
            warnings.warn(f"Model config received the following invalid keys: {', '.join(invalid_keys)}")

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

    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> "ModelConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path).resolve()
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        base_dir = yaml_path.parent
        config_dict = cls._resolve_paths(config_dict, base_dir)
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: PathLike) -> "ModelConfig":
        """Load configuration from JSON file."""
        json_path = Path(json_path).resolve()
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        base_dir = json_path.parent
        config_dict = cls._resolve_paths(config_dict, base_dir)
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[PathLike, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        config_dict = cls._check_fields(config_dict)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: PathLike) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_json(self, json_path: PathLike) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class ModelLoader:
    """Class for loading and initializing TransformerZAM models."""
    
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        """
        Initialize ModelLoader with configuration.
        
        Args:
            config: ModelConfig instance containing model parameters
            device: torch device to load model on (default: cuda if available)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def build_model(self) -> TransformerZAM_multitask:
        """Build model from configuration."""
        model = TransformerZAM_multitask(
            input_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            num_classes_seq2seq=self.config.num_classes_seq2seq,
            num_classes=self.config.num_classes,
            num_heads=self.config.nhead,
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            dim_feedforward=self.config.dim_feedforward,
            dropout_rate_transformer=self.config.dropout,
            positional_encoding_type=self.config.positional_encoding_type,
        )
        
        if self.config.float_dtype == "float64":
            model = model.double()
            
        return model
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, uses config checkpoint_path
                           or searches for best checkpoint in work_dir
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path
            
        if checkpoint_path is None and self.config.work_dir:
            # Search for best checkpoint in work_dir
            best_weights = glob.glob(os.path.join(self.config.work_dir, "best_epoch_*.pth"))
            if best_weights:
                checkpoint_path = best_weights[-1]
            else:
                raise ValueError(f"No checkpoint found in {self.config.work_dir}")
                
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided")
            
        load_weights(self.model, checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    def get_model(self, load_checkpoint: bool = True) -> TransformerZAM_multitask:
        """
        Get initialized model.
        
        Args:
            load_checkpoint: Whether to load checkpoint weights
            
        Returns:
            Initialized TransformerZAM_multitask model
        """
        if self.model is None:
            self.model = self.build_model()
            self.model.to(self.device)
            
        if load_checkpoint:
            self.load_checkpoint()
            
        return self.model
    
    def get_num_parameters(self) -> int:
        """Get total number of model parameters."""
        if self.model is None:
            self.model = self.build_model()
        return sum(p.numel() for p in self.model.parameters())