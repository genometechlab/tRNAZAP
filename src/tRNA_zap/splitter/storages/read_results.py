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
    _logits: Dict[str, np.ndarray]  # Private: e.g., {'seq2seq': array, 'seq_class': array}
    num_chunks: int
    
    # Private access to logits (for internal use)
    @property
    def logits(self) -> Dict[str, np.ndarray]:
        """Get raw logits (internal use only)."""
        return self._logits
    
    @property
    def seq2seq_logits(self) -> Optional[np.ndarray]:
        """Get seq2seq logits (internal use only)."""
        return self._logits.get('seq2seq')
    
    @property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Get classification logits (internal use only)."""
        return self._logits.get('seq_class')
    
    # Public access to probabilities
    @property
    def probs(self) -> Dict[str, np.ndarray]:
        """Get the probabilities for all tasks."""
        from scipy.special import softmax
        return {
            "seq_class": self.classification_probs,
            "seq2seq": self.seq2seq_probs
        }
    
    @property
    def seq2seq_probs(self) -> Optional[np.ndarray]:
        """Get seq2seq probabilities."""
        if 'seq2seq' in self._logits and self._logits['seq2seq'] is not None:
            # Use scipy's softmax for exact match with PyTorch
            from scipy.special import softmax
            return softmax(self._logits['seq2seq'], axis=-1)
        return None
    
    @property
    def classification_probs(self) -> Optional[np.ndarray]:
        """Get classification probabilities."""
        if 'seq_class' in self._logits and self._logits['seq_class'] is not None:
            # Use scipy's softmax for exact match with PyTorch
            from scipy.special import softmax
            return softmax(self._logits['seq_class'], axis=-1)
        return None
    
    # Public access to predictions
    @property
    def preds(self) -> Dict[str, any]:
        """Get the predictions for all tasks."""
        return {
            "seq_class": self.classification_pred,
            "seq2seq": self.seq2seq_preds
        }
    
    @property
    def seq2seq_preds(self) -> Optional[list]:
        """Get seq2seq predictions as a list."""
        if self.seq2seq_probs is not None:
            return np.argmax(self.seq2seq_probs, axis=-1).tolist()
        return None
    
    @property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction as an integer."""
        if self.classification_probs is not None:
            return int(np.argmax(self.classification_probs))
        return None
    
    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self._logits.items()}
        return f"ReadResult(read_id='{self.read_id}', num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"