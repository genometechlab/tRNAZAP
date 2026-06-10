"""
Metadata class for storing inference configuration and settings.
"""
from dataclasses import dataclass, field, replace
from typing import Optional, List
from datetime import datetime


@dataclass
class InferenceMetadata:
    """Metadata for the inference run."""

    # Model configuration
    chunk_size: int
    max_seq_len: int
    model_type: str
    model_name: str
    num_classification_classes: int
    num_segmentation_classes: int

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
                f"num_reads={self.num_reads_processed}, "
                f"chunk_size={self.chunk_size})")

    def copy(self) -> 'InferenceMetadata':
        """
        Return a copy of the metadata.

        Fields are copied via dataclasses.replace (robust to new fields being added),
        and the two mutable containers (label_names, pod5_paths) are duplicated so the
        copy does not share references with the original.
        """
        return replace(
            self,
            label_names=self.label_names.copy() if self.label_names else None,
            pod5_paths=list(self.pod5_paths) if self.pod5_paths else None,
        )