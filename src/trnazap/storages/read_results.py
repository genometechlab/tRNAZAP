"""
ReadResultDetailed class for storing individual read inference results.
"""

import numpy as np
from functools import cached_property
from typing import (
    Union, Optional, Dict, List, Tuple, Protocol, runtime_checkable, TYPE_CHECKING
)
from dataclasses import dataclass
import logging

from scipy.special import softmax  # hoisted from three separate in-method imports

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class ReadResultDetailed:
    """Results for a single read.

    Derived values (logits views, probabilities, predictions, region range) are
    computed lazily and cached. This assumes ``_logits`` is not mutated after
    construction. Treat returned probability/prediction arrays as read-only —
    they are shared across accesses, so mutating one in place would persist.
    """

    read_id: str
    _logits: Dict[str, np.ndarray]  # e.g., {'segmentation': array, 'classification': array, 'frag': array}
    num_chunks: int
    chunk_size: int
    cropped: bool

    def __post_init__(self):
        """Validate logits shapes."""
        if self.segmentation_logits is not None:
            if self.segmentation_logits.ndim != 2:
                raise ValueError(f"segmentation_logits must be 2D, got shape {self.segmentation_logits.shape}")
        if self.classification_logits is not None:
            if self.classification_logits.ndim != 1:
                raise ValueError(f"classification_logits must be 1D, got shape {self.classification_logits.shape}")
        if self.fragmentation_logits is not None:
            if self.fragmentation_logits.ndim != 1:
                raise ValueError(f"fragmentation_logits must be 1D, got shape {self.fragmentation_logits.shape}")

    # ------------------------------------------------------------------------
    # Internal raw logits
    # ------------------------------------------------------------------------

    @property
    def logits(self) -> Dict[str, Optional[np.ndarray]]:
        """Get raw logits. (Fresh dict per call — not cached.)"""
        return {
            "classification": self.classification_logits,
            "segmentation": self.segmentation_logits,
            "fragmentation": self.fragmentation_logits,
        }

    @cached_property
    def fragmentation_logits(self) -> Optional[np.ndarray]:
        """Get fragmentation logits."""
        return self._logits.get('fragmentation')

    @cached_property
    def segmentation_logits(self) -> Optional[np.ndarray]:
        """Get segmentation logits, trimmed by the number of chunks."""
        seg = self._logits.get('segmentation')
        if seg is None:
            return None
        return seg[:self.num_chunks]

    @cached_property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Get classification logits."""
        return self._logits.get('classification')

    # ------------------------------------------------------------------------
    # Probabilities (softmax outputs)
    # ------------------------------------------------------------------------

    @property
    def probs(self) -> Dict[str, Optional[np.ndarray]]:
        """Get probabilities for all tasks. (Fresh dict per call — not cached.)"""
        return {
            "classification": self.classification_probs,
            "segmentation": self.segmentation_probs,
            "fragmentation": self.fragmentation_probs,
        }

    @cached_property
    def segmentation_probs(self) -> Optional[np.ndarray]:
        """Get segmentation probabilities."""
        if self.segmentation_logits is not None:
            return softmax(self.segmentation_logits, axis=-1)
        return None

    @cached_property
    def classification_probs(self) -> Optional[np.ndarray]:
        """Get classification probabilities."""
        if self.classification_logits is not None:
            return softmax(self.classification_logits, axis=-1)
        return None

    @cached_property
    def fragmentation_probs(self) -> Optional[np.ndarray]:
        """Get fragmentation probability."""
        if self.fragmentation_logits is not None:
            return softmax(self.fragmentation_logits, axis=-1)
        return None

    # ------------------------------------------------------------------------
    # Predictions (argmax of probabilities)
    # ------------------------------------------------------------------------

    @property
    def preds(self) -> Dict[str, Union[int, np.ndarray, None]]:
        """Get predictions for all tasks. (Fresh dict per call — not cached.)"""
        return {
            "classification": self.classification_pred,
            "segmentation": self.segmentation_preds,
            "fragmentation": self.fragmentation_pred,
        }

    @cached_property
    def segmentation_preds(self) -> Optional[np.ndarray]:
        """Get segmentation predictions as indices."""
        if self.segmentation_logits is not None:
            return np.argmax(self.segmentation_logits, axis=-1)
        return None

    @cached_property
    def classification_pred(self) -> Optional[int]:
        """Get classification prediction (argmax index)."""
        if self.classification_logits is not None:
            return int(np.argmax(self.classification_logits))
        return None

    def topk_classes(self, k: int = 3) -> Optional[List[int]]:
        """Get the top-k classification classes, largest logit first."""
        if self.classification_logits is None:
            return None

        # argsort gives ascending order; take the last k and reverse for descending.
        topk = np.argsort(self.classification_logits)[-k:][::-1]
        return topk.tolist()

    @cached_property
    def fragmentation_pred(self) -> Optional[int]:
        """Get fragmentation prediction (argmax index)."""
        if self.fragmentation_logits is not None:
            return int(np.argmax(self.fragmentation_logits))
        return None

    # ------------------------------------------------------------------------
    # Region detection & smoothing
    # ------------------------------------------------------------------------

    @cached_property
    def variable_region_range(self) -> Tuple[int, int]:
        """Return (start, end) indices for predicted variable region."""
        preds = self.segmentation_preds
        if preds is None:
            return (-1, -1)
        return self._locate_region_of_interest(preds, 0)

    def get_smoothed_segmentation_preds(
        self,
        device: Union["torch.device", str] = 'cpu',
        return_variable_region_range: bool = False
    ) -> Optional[Union[np.ndarray, tuple]]:
        """Apply CRF smoothing to segmentation predictions."""
        if self.segmentation_logits is not None:
            try:
                from ..utils import CRFSmoother  # deferred: heavy optional dependency
                smoother = CRFSmoother(num_tags=4, device=device)
                predictions_smooth = smoother.decode(self.segmentation_logits,
                                                      lengths=[self.num_chunks])
                if return_variable_region_range:
                    region = self._locate_region_of_interest(predictions_smooth, 0)
                    return predictions_smooth, region
                return predictions_smooth
            except ImportError as e:
                logger.warning(f"[Error] Failed to import crf_smoothing: {e}")
            except Exception as e:
                logger.warning(f"[Error] CRF smoothing failed: {e}")
        return None

    def _locate_region_of_interest(self, preds: np.ndarray, region_id: int) -> Tuple[int, int]:
        """Identify the start and end positions of a specific class in predictions."""
        indices = np.where(preds == region_id)[0]
        if indices.size > 0:
            start_ = indices[0].item()
            end_ = indices[-1].item()
            return (start_ * self.chunk_size, (end_ + 1) * self.chunk_size - 1)
        return (-1, -1)

    # ------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------

    def copy(self) -> "ReadResultDetailed":
        return ReadResultDetailed(
            read_id=self.read_id,
            _logits={k: (v.copy() if v is not None else None) for k, v in self._logits.items()},
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
            cropped=self.cropped,
        )

    def to_compressed(self) -> "ReadResult":
        """
        Convert this ReadResultDetailed into a lightweight ReadResult.

        Keeps the top-3 classification classes, the variable-region range derived
        from the raw (argmax) segmentation, and a boolean fragmentation flag.

        Returns:
            ReadResult
        """
        # --- top-k classes from classification logits ---
        if self.classification_logits is not None:
            top3 = np.argsort(self.classification_logits)[-3:][::-1].astype(int)
        else:
            top3 = np.empty((0,), dtype=int)

        # --- variable region from raw (argmax) segmentation ---
        if self.segmentation_logits is not None:
            variable_region = self.variable_region_range
        else:
            variable_region = (-1, -1)

        # --- fragmentation -> boolean flag ---
        if self.fragmentation_pred is not None:
            fragmented = bool(int(self.fragmentation_pred) > 0)
        else:
            fragmented = False

        return ReadResult(
            read_id=self.read_id,
            top3_classes=top3,
            variable_region_range=variable_region,
            fragmented=fragmented,
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
            cropped=self.cropped
        )

    # ------------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        logit_shapes = {
            k: (v.shape if v is not None else None) for k, v in self._logits.items()
        }
        return (
            f"ReadResultDetailed(read_id='{self.read_id}', "
            f"num_chunks={self.num_chunks}, tasks={list(logit_shapes.keys())})"
        )


@dataclass
class ReadResult:
    """Results for a single read (lightweight / compressed)."""

    read_id: str
    top3_classes: np.ndarray
    variable_region_range: Tuple[int, int]
    fragmented: bool
    num_chunks: int
    chunk_size: int
    cropped: bool

    @property
    def classification_pred(self) -> Optional[int]:
        """Top predicted class index (first of top3_classes), or None if none present."""
        if len(self.top3_classes) >= 1:
            return int(self.top3_classes[0])
        return None

    def copy(self) -> "ReadResult":
        return ReadResult(
            read_id=self.read_id,
            top3_classes=self.top3_classes.copy(),
            variable_region_range=self.variable_region_range,
            fragmented=self.fragmented,
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
            cropped=self.cropped,
        )


# ============================================================================
# Shared interface
# ============================================================================

@runtime_checkable
class ReadResultLike(Protocol):
    """Structural interface shared by ReadResultDetailed and ReadResult.

    Type containers/accessors against this so the members BOTH forms share
    autocomplete cleanly without squiggles. To reach members that exist only on
    the detailed form (logits, probs, segmentation_*, topk_classes, ...), narrow
    with ``isinstance(x, ReadResultDetailed)`` — inside that block the IDE shows
    the full detailed surface.

    Note: this is purely a typing aid; neither class inherits from it. Both
    satisfy it automatically by having the members below.
    """

    read_id: str
    num_chunks: int
    chunk_size: int
    cropped: bool

    @property
    def variable_region_range(self) -> Tuple[int, int]: ...

    @property
    def classification_pred(self) -> Optional[int]: ...

    def copy(self) -> "ReadResultLike": ...