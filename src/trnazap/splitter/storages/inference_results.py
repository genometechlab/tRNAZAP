"""
InferenceResults class for storing all inference results with metadata.
"""
from typing import Dict, List, Optional, Iterator, Tuple, Union, Any
from pathlib import Path
import pickle
import json
import numpy as np

from .read_results import ReadResult
from .inference_metadata import InferenceMetadata


class InferenceResults:
    """Container for all inference results with metadata."""
    
    def __init__(self, metadata: InferenceMetadata):
        """Initialize with metadata.
        
        Args:
            metadata: Inference metadata containing model and run configuration
        """
        self.metadata = metadata
        self._results: Dict[str, ReadResult] = {}
    
    def _add_result(self, read_result: ReadResult) -> None:
        """Add a ReadResult object.
        
        Args:
            read_result: ReadResult object to add
        """
        if read_result._back_reference is None:
            read_result._back_reference = self
        self._results[read_result.read_id] = read_result
        self.metadata.num_reads_processed = len(self._results)
    
    def _add(self, read_id: str, logits: Dict[str, np.ndarray], num_chunks: int) -> None:
        """Add a result for a read.
        
        Args:
            read_id: Read identifier
            logits: Dictionary of logit arrays
            num_chunks: Number of chunks processed for this read
        """
        read_result = ReadResult(
            read_id=read_id,
            _logits=logits,
            num_chunks=num_chunks,
            _back_reference = self
        )
        self._add_result(read_result)
    
    def get(self, read_id: str) -> Optional[ReadResult]:
        """Get result for a specific read.
        
        Args:
            read_id: Read identifier
            
        Returns:
            ReadResult if found, None otherwise
        """
        return self._results.get(read_id)
    
    def __getitem__(self, read_id: str) -> ReadResult:
        """Get result using bracket notation.
        
        Args:
            read_id: Read identifier
            
        Returns:
            ReadResult object
            
        Raises:
            KeyError: If read_id not found
        """
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
    
    @property
    def read_ids(self) -> List[str]:
        """Get list of all read IDs."""
        return list(self._results.keys())
    
    @property
    def label_names(self) -> dict:
        """return the label names"""
        if self.metadata.label_names is not None:
            return self.metadata.label_names
        else:
            raise ValueError("Label names are not provided in the config file")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to pickle file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'InferenceResults':
        """Load results from pickle file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded InferenceResults object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of results.
        
        Returns:
            Dictionary with summary statistics
        """
        total_chunks = sum(r.num_chunks for r in self._results.values())
        
        return {
            'num_reads': len(self),
            'total_chunks': total_chunks,
            'chunk_size': self.metadata.chunk_size,
            'model_type': self.metadata.model_type,
            'device': self.metadata.device,
            'timestamp': self.metadata.timestamp,
            'inference_time': self.metadata.total_inference_time,
        }
    
    def __repr__(self) -> str:
        return f"InferenceResults(num_reads={len(self)}, metadata={self.metadata})"