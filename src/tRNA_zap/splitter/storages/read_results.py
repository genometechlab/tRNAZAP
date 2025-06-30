"""
ReadResult class for storing individual read inference results.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class ReadResult:
    """Results for a single read."""
    
    read_id: str
    logits: Dict[str, np.ndarray]  # e.g., {'seq2seq': array, 'classification': array}
    num_chunks: int
    
    @property
    def seq2seq_logits(self) -> Optional[np.ndarray]:
        """Get seq2seq logits if available."""
        return self.logits.get('seq2seq')
    
    @property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Get classification logits if available."""
        return self.logits.get('classification')
    
    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self.logits.items()}
        return f"ReadResult(read_id='{self.read_id}', num_chunks={self.num_chunks}, logits={logit_shapes})"