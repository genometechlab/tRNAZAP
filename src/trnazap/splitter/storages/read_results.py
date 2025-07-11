"""
ReadResult class for storing individual read inference results.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import logging

from ..io import SaveLoadMixin

if TYPE_CHECKING:
    from .inference_results import InferenceResults
    from .inference_metadata import InferenceMetadata

logger = logging.getLogger(__name__)

@dataclass
class ReadResult(SaveLoadMixin):
    """Results for a single read."""

    read_id: str
    _logits: Dict[str, np.ndarray]  # e.g., {'seq2seq': array, 'seq_class': array}
    num_chunks: int
    chunk_size: int

    def __post_init__(self):
        """Validate logits shapes."""
        if self.seq2seq_logits is not None:
            if self.seq2seq_logits.ndim != 2:
                raise ValueError(f"seq2seq_logits must be 2D, got shape {self.seq2seq_logits.shape}")
        if self.classification_logits is not None:
            if self.classification_logits.ndim != 1:
                raise ValueError(f"classification_logits must be 1D, got shape {self.classification_logits.shape}")

    # ------------------------------------------------------------------------
    # Internal raw logits
    # ------------------------------------------------------------------------

    @property
    def logits(self) -> Dict[str, np.ndarray]:
        """Get raw logits (internal use only)."""
        return self._logits

    @property
    def seq2seq_logits(self) -> Optional[np.ndarray]:
        """Get seq2seq logits."""
        return self._logits.get('seq2seq')

    @property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Get classification logits."""
        return self._logits.get('seq_class')

    # ------------------------------------------------------------------------
    # Probabilities (softmax outputs)
    # ------------------------------------------------------------------------

    @property
    def probs(self) -> Dict[str, Optional[np.ndarray]]:
        """Get probabilities for all tasks."""
        return {
            "seq_class": self.classification_probs,
            "seq2seq": self.seq2seq_probs
        }

    @property
    def seq2seq_probs(self) -> Optional[np.ndarray]:
        """Get seq2seq probabilities."""
        if 'seq2seq' in self._logits and self._logits['seq2seq'] is not None:
            from scipy.special import softmax
            return softmax(self._logits['seq2seq'], axis=-1)
        return None

    @property
    def classification_probs(self) -> Optional[np.ndarray]:
        """Get classification probabilities."""
        if 'seq_class' in self._logits and self._logits['seq_class'] is not None:
            from scipy.special import softmax
            return softmax(self._logits['seq_class'], axis=-1)
        return None

    # ------------------------------------------------------------------------
    # Predictions (argmax of probabilities)
    # ------------------------------------------------------------------------

    @property
    def preds(self) ->  Dict[str, Union[int, np.ndarray]]:
        """Get predictions for all tasks."""
        return {
            "seq_class": self.classification_pred,
            "seq2seq": self.seq2seq_preds
        }

    @property
    def seq2seq_preds(self) -> Optional[np.ndarray]:
        """Get seq2seq predictions as indices."""
        if self.seq2seq_logits is not None:
            return np.argmax(self.seq2seq_logits, axis=-1)
        return None

    @property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction (argmax index)."""
        if self.classification_logits is not None:
            return int(np.argmax(self.classification_logits))
        return None

    # ------------------------------------------------------------------------
    # Region detection & smoothing
    # ------------------------------------------------------------------------

    @property
    def variable_region_range(self) -> tuple:
        """Return (start, end) indices for predicted variable region."""
        preds = self.seq2seq_preds
        return self._locate_region_of_interest(preds, 0)

    def get_smoothed_seq2seq_preds(
        self,
        device: Union[torch.device, str] = 'cpu',
        return_variable_region_range: bool = False
    ) -> Optional[Union[np.ndarray, tuple]]:
        """Apply CRF smoothing to seq2seq predictions."""
        if self.seq2seq_probs is not None:
            try:
                from ..utils import crf_smoothing
                predictions_smooth = crf_smoothing(self.seq2seq_logits, device=device)
                if return_variable_region_range:
                    region = self._locate_region_of_interest(predictions_smooth, 0)
                    return predictions_smooth, region
                return predictions_smooth
            except ImportError as e:
                logger.warning(f"[WARNING] Failed to import crf_smoothing: {e}")
            except Exception as e:
                logger.warning(f"[WARNING] CRF smoothing failed: {e}")

    def _locate_region_of_interest(self, preds: np.ndarray, region_id: int) -> tuple:
        """Identify the start and end positions of a specific class in predictions."""
        indices = np.where(preds == region_id)[0]
        if indices.size > 0:
            start_ = indices[0].item()
            end_ = indices[-1].item()
            return (start_*self.chunk_size, end_*self.chunk_size)
        return (-1, -1)


    # ------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------

    def copy(self) -> "ReadResult":
        return ReadResult(
            read_id=self.read_id,
            _logits={k: v.copy() for k, v in self._logits.items()},
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size
        )
    
    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------
    
    def _to_parquet_records(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """Convert to records for Parquet storage."""
        import numpy as np
        
        seq2seq_flat = self.seq2seq_logits.flatten() if self.seq2seq_logits is not None else np.array([])
        classification_flat = self.classification_logits.flatten() if self.classification_logits is not None else np.array([])
        
        record = {
            'read_id': self.read_id,
            'seq2seq_flat': seq2seq_flat,
            'seq2seq_shape': list(self.seq2seq_logits.shape) if self.seq2seq_logits is not None else [],
            'classification_flat': classification_flat,
            'classification_shape': list(self.classification_logits.shape) if self.classification_logits is not None else [],
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size
        }
        
        metadata = {'format_version': '1.0'}
        return [record], metadata
    
    @classmethod
    def _from_parquet_records(cls, records: List[Dict], metadata: Dict[str, Any]) -> "ReadResult":
        """Reconstruct from Parquet records."""
        import numpy as np
        
        record = records[0]
        logits = {}
        
        if record['seq2seq_shape']:
            logits['seq2seq'] = np.array(record['seq2seq_flat']).reshape(record['seq2seq_shape'])
        if record['classification_shape']:
            logits['seq_class'] = np.array(record['classification_flat']).reshape(record['classification_shape'])
        
        return cls(
            read_id=record['read_id'],
            _logits=logits,
            num_chunks=record['num_chunks'],
            chunk_size=record['chunk_size']
        )

    # ------------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self._logits.items()}
        return (
            f"ReadResult(read_id='{self.read_id}', "
            f"num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"
        )