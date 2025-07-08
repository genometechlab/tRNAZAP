"""
ReadResult class for storing individual read inference results.
"""
import torch
from typing import Union
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
            return np.argmax(self.seq2seq_probs, axis=-1)
        return None
    
    def get_smoothed_seq2seq_preds(self, 
                                   device: Union[torch.device, str] = 'cpu', 
                                   return_variable_region_range: bool = False) -> Optional[list]:
        if self.seq2seq_probs is not None:
            try:
                from ..utils import crf_smoothing
                predictions_smooth = crf_smoothing(self.seq2seq_logits, device=device)
                if return_variable_region_range:
                    range_ = self._locate_region_of_interest(predictions_smooth, 0)
                    return predictions_smooth, range_
                else:
                    return predictions_smooth
            except ImportError as e:
                print(f"[WARNING] Failed to import crf_smoothing: {e}")
            except Exception as e:
                print(f"[WARNING] CRF smoothing failed: {e}")
    
    @property
    def variable_region_range(self):
        """return the first and last tokens predicted as variale region"""
        preds = self.seq2seq_preds
        return self._locate_region_of_interest(preds, 0)
    
    @property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction as an integer."""
        if self.classification_probs is not None:
            return int(np.argmax(self.classification_probs))
        return None
    
    @staticmethod
    def _locate_region_of_interest(preds, region_id):
        indices = np.where(preds == region_id)[0]
        if indices.size > 0:
            return (indices[0].item(), indices[-1].item())
        else:
            return (-1, -1)
    
    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self._logits.items()}
        return f"ReadResult(read_id='{self.read_id}', num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"