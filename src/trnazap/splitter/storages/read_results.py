"""
ReadResult class with dynamic property generation for arbitrary logit keys.
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


def _create_logits_property(key: str):
    """Factory function to create a logits property getter for a specific key."""
    def getter(self) -> Optional[np.ndarray]:
        if key not in self._logits:
            raise KeyError(f"The model did not output {key} logits")
        logits = self._logits[key]
        # Apply trimming for seq2seq-like outputs (2D arrays)
        if logits.ndim == 2 and hasattr(self, 'num_chunks'):
            return logits[:self.num_chunks]
        return logits
    return property(getter)


def _create_probs_property(key: str):
    """Factory function to create a probs property getter for a specific key."""
    def getter(self) -> Optional[np.ndarray]:
        if key in self._logits and self._logits[key] is not None:
            from scipy.special import softmax
            return softmax(self._logits[key], axis=-1)
        return None
    return property(getter)


def _create_preds_property(key: str):
    """Factory function to create a predictions property getter for a specific key."""
    def getter(self) -> Optional[Union[int, np.ndarray]]:
        if key in self._logits and self._logits[key] is not None:
            preds = np.argmax(self._logits[key], axis=-1)
            # Return int for 1D outputs (classification), array for 2D (seq2seq)
            if self._logits[key].ndim == 1:
                return int(preds)
            return preds
        return None
    return property(getter)


class DynamicPropertiesMeta(type):
    """Metaclass that dynamically creates properties based on logit keys."""
    
    def __call__(cls, *args, **kwargs):
        # Create the instance
        instance = super().__call__(*args, **kwargs)
        
        # Dynamically add properties based on _logits keys
        for key in instance._logits.keys():
            # Create property names
            logits_prop = f"{key}_logits"
            probs_prop = f"{key}_probs"
            preds_prop = f"{key}_preds"
            
            # Add properties to the instance's class
            if not hasattr(instance.__class__, logits_prop):
                setattr(instance.__class__, logits_prop, _create_logits_property(key))
            if not hasattr(instance.__class__, probs_prop):
                setattr(instance.__class__, probs_prop, _create_probs_property(key))
            if not hasattr(instance.__class__, preds_prop):
                setattr(instance.__class__, preds_prop, _create_preds_property(key))
        
        return instance


@dataclass
class ReadResult(SaveLoadMixin, metaclass=DynamicPropertiesMeta):
    """Results for a single read with dynamic property generation."""

    read_id: str
    _logits: Dict[str, np.ndarray]  # Can have any keys
    num_chunks: int
    chunk_size: int

    def __post_init__(self):
        """Validate logits shapes based on common patterns."""
        for key, logits in self._logits.items():
            if logits.ndim not in [1, 2]:
                raise ValueError(f"{key} logits must be 1D or 2D, got shape {logits.shape}")

    # ------------------------------------------------------------------------
    # Aggregate properties that work with any keys
    # ------------------------------------------------------------------------

    @property
    def logits(self) -> Dict[str, np.ndarray]:
        """Get all raw logits."""
        result = {}
        for key in self._logits:
            # Apply trimming for 2D arrays
            if self._logits[key].ndim == 2:
                result[key] = self._logits[key][:self.num_chunks]
            else:
                result[key] = self._logits[key]
        return result

    @property
    def probs(self) -> Dict[str, Optional[np.ndarray]]:
        """Get probabilities for all tasks."""
        from scipy.special import softmax
        return {
            key: softmax(logits, axis=-1) if logits is not None else None
            for key, logits in self.logits.items()
        }

    @property
    def preds(self) -> Dict[str, Union[int, np.ndarray]]:
        """Get predictions for all tasks."""
        result = {}
        for key, logits in self.logits.items():
            if logits is not None:
                preds = np.argmax(logits, axis=-1)
                # Return int for 1D outputs, array for 2D
                result[key] = int(preds) if logits.ndim == 1 else preds
            else:
                result[key] = None
        return result

    # ------------------------------------------------------------------------
    # Backwards compatibility properties
    # ------------------------------------------------------------------------

    @property
    def seq2seq_logits(self) -> Optional[np.ndarray]:
        """Backwards compatibility for seq2seq_logits."""
        if hasattr(self, '_seq2seq_logits_cached'):
            return self._seq2seq_logits_cached
        return self._logits.get('seq2seq', None)

    @property
    def classification_logits(self) -> Optional[np.ndarray]:
        """Backwards compatibility for classification_logits."""
        if hasattr(self, '_seq_class_logits_cached'):
            return self._seq_class_logits_cached
        return self._logits.get('seq_class', None)

    @property
    def seq2seq_probs(self) -> Optional[np.ndarray]:
        """Backwards compatibility for seq2seq_probs."""
        if hasattr(self, '_seq2seq_probs_cached'):
            return self._seq2seq_probs_cached
        return self.probs.get('seq2seq', None)

    @property
    def classification_probs(self) -> Optional[np.ndarray]:
        """Backwards compatibility for classification_probs."""
        if hasattr(self, '_seq_class_probs_cached'):
            return self._seq_class_probs_cached
        return self.probs.get('seq_class', None)

    @property
    def seq2seq_preds(self) -> Optional[np.ndarray]:
        """Backwards compatibility for seq2seq_preds."""
        if hasattr(self, '_seq2seq_preds_cached'):
            return self._seq2seq_preds_cached
        return self.preds.get('seq2seq', None)

    @property
    def classification_pred(self) -> Optional[int]:
        """Backwards compatibility for classification_pred."""
        if hasattr(self, '_seq_class_preds_cached'):
            return self._seq_class_preds_cached
        return self.preds.get('seq_class', None)

    # ------------------------------------------------------------------------
    # Region detection & smoothing
    # ------------------------------------------------------------------------

    @property
    def variable_region_range(self) -> tuple:
        """Return (start, end) indices for predicted variable region."""
        preds = self.segmentation_preds
        return self._locate_region_of_interest(preds, 0)

    def get_smoothed_segments(
        self,
        device: Union[torch.device, str] = 'cpu',
        return_variable_region_range: bool = False
    ) -> Optional[Union[np.ndarray, tuple]]:
        """Apply CRF smoothing to segmentation predictions."""
        if self.segmentation_logits is not None:
            try:
                from ..utils import crf_smoothing
                predictions_smooth = crf_smoothing(self.segmentation_logits, device=device)
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
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available task keys."""
        return list(self._logits.keys())
    
    # ------------------------------------------------------------------------
    # I/O - Updated to handle arbitrary keys
    # ------------------------------------------------------------------------
    
    def _to_parquet_records(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """Convert to records for Parquet storage."""
        record = {
            'read_id': self.read_id,
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size,
            'logit_keys': list(self._logits.keys())
        }
        
        # Flatten each logit array and store with shape
        for key, logits in self._logits.items():
            record[f'{key}_flat'] = logits.flatten()
            record[f'{key}_shape'] = list(logits.shape)
        
        metadata = {'format_version': '2.0'}  # Updated version
        return [record], metadata
    
    @classmethod
    def _from_parquet_records(cls, records: List[Dict], metadata: Dict[str, Any]) -> "ReadResult":
        """Reconstruct from Parquet records."""
        import numpy as np
        
        record = records[0]
        logits = {}
        
        # Reconstruct logits from flattened arrays
        for key in record['logit_keys']:
            flat_key = f'{key}_flat'
            shape_key = f'{key}_shape'
            if flat_key in record and shape_key in record:
                logits[key] = np.array(record[flat_key]).reshape(record[shape_key])
        
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