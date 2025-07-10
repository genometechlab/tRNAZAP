"""
InferenceResults class for storing all inference results with metadata.
"""

from typing import Dict, List, Optional, Iterator, Tuple, Union, Any
from pathlib import Path
import pickle
import numpy as np
import logging

from .read_results import ReadResult
from .inference_metadata import InferenceMetadata

logger = logging.getLogger(__name__)

# =============================================================================
# InferenceResults Container
# =============================================================================

class InferenceResults:
    """Container for all inference results with metadata."""

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, metadata: InferenceMetadata):
        """
        Initialize with metadata.

        Args:
            metadata: Inference metadata containing model and run configuration
        """
        self.metadata = metadata
        self._results: Dict[str, ReadResult] = {}

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    def _add_result(self, read_result: ReadResult) -> None:
        """
        Add a ReadResult object.

        Args:
            read_result: ReadResult object to add
        """
        self._results[read_result.read_id] = read_result
        self.metadata.num_reads_processed = len(self._results)

    def _add(self, read_id: str, logits: Dict[str, np.ndarray], num_chunks: int) -> None:
        """
        Add a result for a read.

        Args:
            read_id: Read identifier
            logits: Dictionary of logit arrays
            num_chunks: Number of chunks processed for this read
        """
        read_result = ReadResult(
            read_id=read_id,
            _logits=logits,
            num_chunks=num_chunks,
            chunk_size=self.metadata.chunk_size
        )
        self._add_result(read_result)

    def get(self, read_id: str) -> Optional[ReadResult]:
        """
        Get result for a specific read.

        Args:
            read_id: Read identifier

        Returns:
            ReadResult if found, None otherwise
        """
        return self._results.get(read_id)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def read_ids(self) -> List[str]:
        """Get list of all read IDs."""
        return list(self._results.keys())

    @property
    def label_names(self) -> dict:
        """Return the label names."""
        if self.metadata.label_names is not None:
            return self.metadata.label_names
        raise ValueError("Label names are not provided in the config file")

    # -------------------------------------------------------------------------
    # Iterators and Access
    # -------------------------------------------------------------------------

    def __getitem__(self, read_id: str) -> ReadResult:
        """Get result using bracket notation."""
        if read_id not in self._results:
            raise KeyError(f"No results found for read_id: {read_id}")
        return self._results[read_id]

    def __contains__(self, read_id: str) -> bool:
        """Check if read_id exists in results."""
        return read_id in self._results

    def __len__(self) -> int:
        """Get number of reads in results."""
        return len(self._results)

    def __iter__(self) -> Iterator[str]:
        """Iterate over read IDs."""
        return iter(self._results)

    def items(self) -> Iterator[Tuple[str, ReadResult]]:
        """Iterate over (read_id, ReadResult) pairs."""
        return self._results.items()

    def values(self) -> Iterator[ReadResult]:
        """Iterate over ReadResult objects."""
        return self._results.values()

    def keys(self) -> List[str]:
        """Get all read IDs."""
        return list(self._results.keys())

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save results to pickle file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, paths: Union[str, Path, List[Union[str, Path]]]) -> "InferenceResults":
        """
        Load one or more InferenceResults from pickle file(s).

        Args:
            paths: Path or list of paths to .pkl files.

        Returns:
            A single (merged if needed) InferenceResults instance.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        loaded_results = []

        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            with open(path, 'rb') as f:
                result = pickle.load(f)
            
            if not isinstance(result, cls):
                raise TypeError(f"File '{path}' does not contain an InferenceResults object.")

            loaded_results.append(result)

        if len(loaded_results) == 1:
            return loaded_results[0]
        elif len(loaded_results)>1:
            return cls.merge(*loaded_results)
        else:
            raise ValueError(f"No Path is provided")



    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of results.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'num_reads': len(self),
            'chunk_size': self.metadata.chunk_size,
            'model_name': self.metadata.model_name,
            'inference_time': self.metadata.total_inference_time,
        }

    # -------------------------------------------------------------------------
    # Dunder Methods
    # -------------------------------------------------------------------------

    @classmethod
    def merge(cls, *results: "InferenceResults") -> "InferenceResults":
        """
        Merge multiple InferenceResults instances into one.

        Args:
            *results: One or more InferenceResults objects.

        Returns:
            A single merged InferenceResults instance.

        Notes:
            - Assumes all metadata are compatible (model name, chunk size, max_seq_len).
            - Skips duplicate read IDs with a warning.
        """
        if not results:
            raise ValueError("At least one InferenceResults instance must be provided.")

        base = results[0]

        for other in results[1:]:
            if not isinstance(other, InferenceResults):
                raise TypeError("All inputs must be instances of InferenceResults.")

            # Check metadata compatibility
            if base.metadata.model_name != other.metadata.model_name:
                raise ValueError("Model names do not match.")
            if base.metadata.chunk_size != other.metadata.chunk_size:
                raise ValueError("Chunk sizes do not match.")
            if base.metadata.max_seq_len != other.metadata.max_seq_len:
                raise ValueError("Max sequence lengths do not match.")

        # Create combined result
        merged = InferenceResults(metadata=base.metadata.copy())

        for result in results:
            for read_id, read_result in result.items():
                if read_id in merged:
                    logger.warning(f"Duplicate read_id '{read_id}' skipped during merge.")
                    continue
                merged._add_result(read_result.copy())

        return merged


    def __repr__(self) -> str:
        return f"InferenceResults(num_reads={len(self)}, metadata={self.metadata})"
