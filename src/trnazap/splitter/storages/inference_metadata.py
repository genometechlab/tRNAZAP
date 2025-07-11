"""
Metadata class for storing inference configuration and settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Set
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
    pod5_paths: Optional[List[str]] = None
    num_reads_processed: int = 0
    total_inference_time: Optional[float] = None
    
    def __repr__(self) -> str:
        return (f"InferenceMetadata(model_name='{self.model_name}', "
                f"num_reads={self.num_reads_processed})"
                f"chunk_size='{self.chunk_size}', ")

    def copy(self) -> 'InferenceMetadata':
        """Return a deep copy of the metadata."""
        return InferenceMetadata(
            chunk_size=self.chunk_size,
            max_seq_len=self.max_seq_len,
            model_type=self.model_type,
            model_name=self.model_name,
            num_classes=self.num_classes,
            num_classes_seq2seq=self.num_classes_seq2seq,
            label_names=self.label_names.copy() if self.label_names else None,
            batch_size=self.batch_size,
            device=self.device,
            float_dtype=self.float_dtype,
            timestamp=self.timestamp,
            model_checkpoint=self.model_checkpoint,
            pod5_paths=list(self.pod5_paths) if self.pod5_paths else None,
            num_reads_processed=self.num_reads_processed,
            total_inference_time=self.total_inference_time
        )