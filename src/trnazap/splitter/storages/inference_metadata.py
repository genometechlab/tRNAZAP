"""
Metadata class for storing inference configuration and settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from pathlib import Path


@dataclass
class InferenceMetadata:
    """Metadata for the inference run."""
    
    # Model configuration
    chunk_size: int
    max_seq_len: int
    model_type: str
    model_name: str
    num_classes: int
    num_classes_seq2seq: int

    # Label names
    label_names: dict
    
    # Inference settings
    batch_size: int
    device: str
    float_dtype: str
    
    # Run information
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_checkpoint: Optional[str] = None
    pod5_paths: Optional[List[Path]] = None
    num_reads_processed: int = 0
    total_inference_time: Optional[float] = None
    
    def __repr__(self) -> str:
        return (f"InferenceMetadata(chunk_size={self.chunk_size}, "
                f"model_type='{self.model_type}', "
                f"device='{self.device}', "
                f"num_reads={self.num_reads_processed})")