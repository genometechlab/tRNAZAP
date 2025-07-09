"""
ReadResult class for storing individual read inference results.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .inference_results import InferenceResults
    from .inference_metadata import InferenceMetadata


@dataclass
class ReadResult:
    """Results for a single read."""

    read_id: str
    _logits: Dict[str, np.ndarray]  # e.g., {'seq2seq': array, 'seq_class': array}
    num_chunks: int
    _back_reference: Optional["InferenceResults"] = None  # Reference to parent InferenceResults

    # ------------------------------------------------------------------------
    # Internal metadata access
    # ------------------------------------------------------------------------

    @property
    def metadata(self) -> Optional["InferenceMetadata"]:
        """Access parent metadata via back-reference."""
        if self._back_reference:
            return self._back_reference.metadata
        raise ValueError("Reference to results container not found.")

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
    def probs(self) -> Dict[str, np.ndarray]:
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
    def preds(self) -> Dict[str, any]:
        """Get predictions for all tasks."""
        return {
            "seq_class": self.classification_pred,
            "seq2seq": self.seq2seq_preds
        }

    @property
    def seq2seq_preds(self) -> Optional[np.ndarray]:
        """Get seq2seq predictions as indices."""
        if self.seq2seq_probs is not None:
            return np.argmax(self.seq2seq_probs, axis=-1)
        return None

    @property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction (argmax index)."""
        if self.classification_probs is not None:
            return int(np.argmax(self.classification_probs))
        return None

    @property
    def classification_pred_cls(self) -> Optional[str]:
        """Get classification prediction label using metadata."""
        try:
            lbl_ind = self.classification_pred
            return self.metadata.label_names[lbl_ind]
        except Exception as e:
            raise e

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
                print(f"[WARNING] Failed to import crf_smoothing: {e}")
            except Exception as e:
                print(f"[WARNING] CRF smoothing failed: {e}")

    @staticmethod
    def _locate_region_of_interest(preds, region_id: int) -> tuple:
        """Identify the start and end positions of a specific class in predictions."""
        indices = np.where(preds == region_id)[0]
        if indices.size > 0:
            return (indices[0].item(), indices[-1].item())
        return (-1, -1)

    # ------------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self._logits.items()}
        return (
            f"ReadResult(read_id='{self.read_id}', "
            f"num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"
        )
