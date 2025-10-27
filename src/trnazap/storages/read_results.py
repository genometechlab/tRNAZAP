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

if TYPE_CHECKING:
    from .inference_results import InferenceResults
    from .inference_metadata import InferenceMetadata

logger = logging.getLogger(__name__)

@dataclass
class ReadResult:
    """Results for a single read."""

    read_id: str
    _logits: Dict[str, np.ndarray]  # e.g., {'segmentation': array, 'classification': array, 'frag': array}
    num_chunks: int
    chunk_size: int

    def __post_init__(self):
        """Validate logits shapes."""
        if self.segmentation_logits is not None:
            if self.segmentation_logits.ndim != 2:
                raise ValueError(f"segmentation_logits must be 2D, got shape {self.segmentation_logits.shape}")
        if self.classification_logits is not None:
            if self.classification_logits.ndim != 1:
                raise ValueError(f"classification_logits must be 1D, got shape {self.classification_logits.shape}")
        # Add this:
        if self.fragmentation_logits is not None:
            if self.fragmentation_logits.ndim != 1:
                raise ValueError(f"fragmentation_logits must be 1D, got shape {self.fragmentation_logits.shape}")


    # ------------------------------------------------------------------------
    # Internal raw logits
    # ------------------------------------------------------------------------

    @property
    def logits(self) -> Dict[str, np.ndarray]:
        """Get raw logits."""
        return {
            "classification": self.classification_logits,
            "segmentation": self.segmentation_logits,
            "fragmentation": self.fragmentation_logits,
        }
        
    @property
    def fragmentation_logits(self) -> Optional[np.ndarray]:
        """Get fragmentation logits"""
        if "fragmentation" in self._logits:
            return self._logits.get('fragmentation')
        else:
            return None

    @property
    def segmentation_logits(self) -> Optional[np.ndarray]:
        """Get segmentation logits, trimmed by the number of chunks."""
        if "segmentation" in self._logits:
            return self._logits.get('segmentation')[:self.num_chunks]
        else:
            return None

    @property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Get classification logits."""
        if "classification" in self._logits:
            return self._logits.get('classification')
        else:
            return None

    # ------------------------------------------------------------------------
    # Probabilities (softmax outputs)
    # ------------------------------------------------------------------------

    @property
    def probs(self) -> Dict[str, Optional[np.ndarray]]:
        """Get probabilities for all tasks."""
        return {
            "classification": self.classification_probs,
            "segmentation": self.segmentation_probs,
            "fragmentation": self.fragmentation_probs,
        }

    @property
    def segmentation_probs(self) -> Optional[np.ndarray]:
        """Get segmentation probabilities."""
        if 'segmentation' in self._logits and self._logits['segmentation'] is not None:
            from scipy.special import softmax
            return softmax(self.segmentation_logits, axis=-1)
        return None

    @property
    def classification_probs(self) -> Optional[np.ndarray]:
        """Get classification probabilities."""
        if 'classification' in self._logits and self._logits['classification'] is not None:
            from scipy.special import softmax
            return softmax(self.classification_logits, axis=-1)
        return None
    
    @property
    def fragmentation_probs(self) -> Optional[np.ndarray]:
        """Get fragmentation probability."""
        if 'fragmentation' in self._logits and self._logits['fragmentation'] is not None:
            from scipy.special import softmax
            return softmax(self.fragmentation_logits, axis=-1)
        return None

    # ------------------------------------------------------------------------
    # Predictions (argmax of probabilities)
    # ------------------------------------------------------------------------

    @property
    def preds(self) ->  Dict[str, Union[int, np.ndarray]]:
        """Get predictions for all tasks."""
        return {
            "classification": self.classification_pred,
            "segmentation": self.segmentation_preds,
            "fragmentation": self.fragmentation_pred,
        }

    @property
    def segmentation_preds(self) -> Optional[np.ndarray]:
        """Get segmentation predictions as indices."""
        if self.segmentation_logits is not None:
            return np.argmax(self.segmentation_logits, axis=-1)
        return None

    @property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction (argmax index)."""
        if self.classification_logits is not None:
            return int(np.argmax(self.classification_logits))
        return None
    
    @property
    def topk_classes(self, k: int = 3) -> Optional[List[int]]:
        """Get Top 3 prediction classes"""
        if self.classification_logits is None:
            return None

        # Use argsort to get indices of top k logits (largest first)
        topk = np.argsort(self.classification_logits)[-k:][::-1]
        return topk.tolist()
    
    @property
    def fragmentation_pred(self) -> Optional[int]:
        """Get fragmentation prediction (argmax index)."""
        if self.fragmentation_logits is not None:
            return int(np.argmax(self.fragmentation_logits))
        return None

    # ------------------------------------------------------------------------
    # Region detection & smoothing
    # ------------------------------------------------------------------------

    @property
    def variable_region_range(self) -> tuple:
        """Return (start, end) indices for predicted variable region."""
        preds = self.segmentation_preds
        if preds is None:
            return (-1, -1)
        return self._locate_region_of_interest(preds, 0)

    def get_smoothed_segmentation_preds(
        self,
        device: Union[torch.device, str] = 'cpu',
        return_variable_region_range: bool = False
    ) -> Optional[Union[np.ndarray, tuple]]:
        """Apply CRF smoothing to segmentation predictions."""
        if self.segmentation_probs is not None:
            try:
                from ..utils import crf_smoothing
                predictions_smooth = crf_smoothing(self.segmentation_logits, lengths=[self.num_chunks], device=device)
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
            return (start_*self.chunk_size, (end_+1)*self.chunk_size-1)
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
        
    def to_compressed(
        self,
        *,
        k: int = 3,
        device: Union[torch.device, str] = "cpu",
        smoothed_preds: Optional[np.ndarray] = None
    ) -> "ReadResultCompressed":
        """
        Convert this ReadResult into a lightweight ReadResultCompressed.

        Args:
            k: how many top classes to keep from classification logits (default: 3)
            device: device for CRF smoothing if used (default: 'cpu')
            smoothed_preds: smoothed prediction using CRF. if not provided, will be computed

        Returns:
            ReadResultCompressed
        """
        # --- top-k classes from classification logits ---
        if self.classification_logits is not None:
            topk = np.argsort(self.classification_logits)[-k:][::-1].astype(int)
        else:
            topk = np.empty((0,), dtype=int)

        # --- variable region from raw (argmax) segmentation ---
        if self.segmentation_logits is not None:
            variable_region = self.variable_region_range
        else:
            variable_region = (-1, -1)
            
        if smoothed_preds is not None:
            smoothed_variable_region = self._locate_region_of_interest(smoothed_preds, 0)
        else:
            _, smoothed_variable_region = self.get_smoothed_segmentation_preds(device, True)

        # --- fragmentation -> boolean flag ---
        if self.fragmentation_pred is not None:
            fragmented = bool(int(self.fragmentation_pred) > 0)
        else:
            fragmented = False

        return ReadResultCompressed(
            read_id=self.read_id,
            top3_classes=topk,
            variable_region_range=variable_region,
            smoothed_variable_region_range=smoothed_variable_region,
            fragmented=fragmented,
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
        )

    
    # ------------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        logit_shapes = {k: v.shape for k, v in self._logits.items()}
        return (
            f"ReadResult(read_id='{self.read_id}', "
            f"num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"
        )
        
        
@dataclass
class ReadResultCompressed:
    """Results for a single read."""

    read_id: str
    top3_classes: np.ndarray
    variable_region_range: Tuple
    smoothed_variable_region_range: Tuple
    fragmented: bool
    num_chunks: int
    chunk_size: int