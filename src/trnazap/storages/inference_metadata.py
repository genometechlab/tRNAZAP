"""
Metadata class for storing inference configuration and settings.
"""
from dataclasses import dataclass, field, fields, replace
from typing import Optional, List
from datetime import datetime


@dataclass
class InferenceMetadata:
    """Metadata for the inference run."""

    # Model configuration
    chunk_size: int
    max_seq_len: int
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
    pod5_paths: Optional[List[str]] = None
    num_reads_processed: int = 0
    total_inference_time: Optional[float] = None

    # Provenance (format >= 102; None when read from older archives)
    config_name: Optional[str] = None              # config file name; None for dict/object configs
    tasks: Optional[List[str]] = None              # tasks the model predicts
    head_type: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    run_id: Optional[str] = None                   # shared by every shard of one run
    trnazap_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceMetadata":
        """Build from a header dict, ignoring keys this version doesn't know.

        Newer writers may add fields, and older ones wrote some that were since
        dropped (e.g. ``model_type``); neither should make an archive unreadable.
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def __repr__(self) -> str:
        return (f"InferenceMetadata(model_name='{self.model_name}', "
                f"num_reads={self.num_reads_processed}, "
                f"chunk_size={self.chunk_size})")

    def copy(self) -> 'InferenceMetadata':
        """
        Return a copy of the metadata.

        Fields are copied via dataclasses.replace (robust to new fields being added),
        and the mutable containers (label_names, pod5_paths, tasks) are duplicated so the
        copy does not share references with the original.
        """
        return replace(
            self,
            label_names=self.label_names.copy() if self.label_names else None,
            pod5_paths=list(self.pod5_paths) if self.pod5_paths else None,
            tasks=list(self.tasks) if self.tasks else self.tasks,
        )