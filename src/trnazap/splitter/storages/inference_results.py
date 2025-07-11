"""
InferenceResults class for storing all inference results with metadata.
"""

from typing import Dict, List, Optional, Iterator, Tuple, Union, Any
from pathlib import Path
import pickle
import numpy as np
import logging

import h5py
import json

from .read_results import ReadResult
from .inference_metadata import InferenceMetadata
from ..io import MultiLoadMixin
from ..utils import PathSet

logger = logging.getLogger(__name__)

# =============================================================================
# InferenceResults Container
# =============================================================================

class InferenceResults(MultiLoadMixin):
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
    # I/O Methods
    # -------------------------------------------------------------------------
    
    def _to_parquet_records(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """Convert to records for Parquet storage."""
        from dataclasses import asdict
        
        records = []
        for read_id, result in self._results.items():
            result_records, _ = result._to_parquet_records()
            records.extend(result_records)
        
        metadata_dict = asdict(self.metadata)
        if metadata_dict.get('pod5_paths') is not None:
            metadata_dict['pod5_paths'] = list(metadata_dict['pod5_paths'])
        
        metadata = {
            'inference_metadata': metadata_dict,
            'format_version': '2.0',
            'num_results': len(self._results)
        }
        
        return records, metadata
    
    @classmethod
    def _from_parquet_records(cls, records: List[Dict], metadata: Dict[str, Any]) -> "InferenceResults":
        """Reconstruct from Parquet records."""
        metadata_dict = metadata['inference_metadata']
        if metadata_dict.get('pod5_paths') is not None:
            metadata_dict['pod5_paths'] = list(metadata_dict['pod5_paths'])
        
        inference_metadata = InferenceMetadata(**metadata_dict)
        result = cls(metadata=inference_metadata)
        
        for record in records:
            read_result = ReadResult._from_parquet_records([record], {})
            result._add_result(read_result)
        
        return result

    # -------------------------------------------------------------------------
    # Utility
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
        merged_pod5_pths = PathSet()

        for result in results:
            merged_pod5_pths = merged_pod5_pths + PathSet.from_list(result.metadata.pod5_paths)
            for read_id, read_result in result.items():
                if read_id in merged:
                    logger.warning(f"Duplicate read_id '{read_id}' skipped during merge.")
                    continue
                merged._add_result(read_result.copy())

        merged.metadata.pod5_paths = merged_pod5_pths.to_list()

        return merged
    
    def __add__(self, other: "InferenceResults"):
        return self.merge(self, other)

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

    def __repr__(self) -> str:
        return f"InferenceResults(num_reads={len(self)}, metadata={self.metadata})"