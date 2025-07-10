"""
InferenceResults class for storing all inference results with metadata.
"""

from typing import Dict, List, Optional, Iterator, Tuple, Union, Any
from pathlib import Path
import pickle
import numpy as np

from .read_results import ReadResult
from .inference_metadata import InferenceMetadata


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
        if read_result._back_reference is None:
            read_result._back_reference = self
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
            _back_reference=self
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
    def load(cls, paths: Union[str, Path, List[Union[str, Path]]]) -> 'InferenceResults':
        """
        Load one or more InferenceResults from pickle file(s).

        Args:
            paths: Path or list of paths to .pkl files

        Returns:
            A single InferenceResults instance (merged if multiple files)
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]  # Normalize to list

        combined: Optional[InferenceResults] = None

        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, 'rb') as f:
                loaded = pickle.load(f)
                if not isinstance(loaded, cls):
                    raise TypeError(f"File {path} does not contain InferenceResults.")

                if combined is None:
                    combined = loaded
                else:
                    combined = combined + loaded

        if combined is None:
            raise ValueError("No valid InferenceResults loaded.")

        return combined

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

    def __add__(self, other: "InferenceResults") -> "InferenceResults":
        """
        Combine two InferenceResults objects.

        Args:
            other: Another InferenceResults instance

        Returns:
            Combined InferenceResults

        Notes:
            - Skips duplicate read IDs with a warning.
            - Ensures metadata compatibility (model name, chunk size, max seq len).
        """
        if not isinstance(other, InferenceResults):
            raise ValueError("Argument must be an instance of InferenceResults.")

        if self.metadata.model_name != other.metadata.model_name:
            raise ValueError("Cannot add results from different models.")

        if self.metadata.chunk_size != other.metadata.chunk_size:
            raise ValueError("Chunk sizes do not match.")

        if self.metadata.max_seq_len != other.metadata.max_seq_len:
            raise ValueError("Max sequence lengths do not match.")

        new_metadata = self.metadata.copy() if hasattr(self.metadata, 'copy') else self.metadata
        combined = InferenceResults(metadata=new_metadata)

        # Add self results
        for read_id, result in self.items():
            combined._add_result(result.copy(new_back_reference=combined))

        # Add other's results
        for read_id, result in other.items():
            if read_id in combined:
                print(f"[Warning] Duplicate read_id '{read_id}' skipped.")
                continue
            combined._add_result(result.copy(new_back_reference=combined))

        return combined

    def __repr__(self) -> str:
        return f"InferenceResults(num_reads={len(self)}, metadata={self.metadata})"
