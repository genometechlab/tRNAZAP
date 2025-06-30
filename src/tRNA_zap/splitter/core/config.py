import os
import glob
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json

from ..model import TransformerZAM_multitask
from ..utils import load_weights


@dataclass
class ModelConfig:
    """Configuration class for TransformerZAM model parameters."""
    
    # Model architecture parameters
    chunk_size: int = 4000
    max_seq_len: int = 1000
    num_classes: int = 1
    num_classes_seq2seq: int = 4
    nhead: int = 8
    num_layers: int = 6
    hidden_size: int = 512
    dim_feedforward: int = 2048
    dropout: float = 0.1
    positional_encoding_type: str = "sinusoidal"
    
    # Training parameters
    float_dtype: str = "float32"
    
    # Model checkpoint
    checkpoint_path: Optional[str] = None
    work_dir: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.__dict__
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


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