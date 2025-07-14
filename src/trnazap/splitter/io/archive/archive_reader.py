import struct
import json
import os
import zstandard as zstd
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, TYPE_CHECKING, List, Union, Collection, Set, Generator, Iterable
import numpy as np
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache, partial
from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    RECORD_MARKER
)

from ...utils import search_path

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, ReadResult, InferenceResults

logger = logging.getLogger(__name__)

PathLike = Union[str, Path, os.PathLike]


class ZIRReader:
    """Read inference results from ZIR archive format(s)."""
    
    def __init__(self, paths: Union[Path, List[Path]], index=False):
        """
        Initialize reader.
        
        Args:
            path: Archive file path or list of paths
            index: Whether to build index on initialization
        """     
        self._paths: List[Path] = sorted(
            self._collect_dataset(
                paths, recursive=True, pattern='*.zir', threads=4
            )
        )
        self.is_multi = len(self._paths) > 1

        
        self.files = []
        metadata_dicts = []
        self.decompressors = []
        self.record_counts = []
        self.record_count = 0
        
        for p in self._paths:
            file = open(p, 'rb')
            self.files.append(file)

            metadata_dict, record_count = self._read_header_from_file(file)
            metadata_dicts.append(metadata_dict)
            self.decompressors.append(zstd.ZstdDecompressor())
            self.record_counts.append(record_count)
            self.record_count += record_count
        
        # Check metadata compatibility for multiple files
        if len(metadata_dicts) > 1:
            self._check_metadata_compatibility(metadata_dicts)
        
        self.metadata_dict = metadata_dicts[0]
        
        self._index = None
        self._current_file_idx = 0
        
        # Build index if demanded
        if index:
            self.build_index()

    @staticmethod
    def _collect_dataset(
        paths: Union[PathLike, Collection[PathLike]],
        recursive: bool,
        pattern: str,
        threads: int,
    ) -> Set[Path]:
        if isinstance(paths, (str, Path, os.PathLike)):
            paths = [paths]

        if not isinstance(paths, Collection):
            raise TypeError(
                f"paths must be a Collection[PathOrStr] but found {type(paths)=}"
            )

        paths = [Path(p) for p in paths]
        collected: Set[Path] = set()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            search = partial(search_path, recursive=recursive, patterns=[pattern])
            for coll in executor.map(search, paths):
                collected.update(coll)
        return collected

    def _check_metadata_compatibility(self, metadata_dicts):
        """Check if all files have compatible metadata."""
        first_metadata = metadata_dicts[0]
        
        # Critical fields that must match
        critical_fields = [
            'model_name',
            'model_type', 
            'chunk_size',
            'num_classes',
            'num_classes_seq2seq',
            'max_seq_len'
        ]
        
        for i, metadata in enumerate(metadata_dicts[1:], 1):
            for field in critical_fields:
                first_value = first_metadata.get(field)
                current_value = metadata.get(field)
                
                if first_value != current_value:
                    for f in self.files:
                        f.close()
                    
                    raise ValueError(
                        f"Incompatible metadata in file {self._paths[i]}: "
                        f"{field} mismatch - expected '{first_value}' but got '{current_value}'. "
                        f"All archives must be from the same model."
                    )
        
        warning_fields = ['batch_size', 'device', 'float_dtype']
        for field in warning_fields:
            values = [m.get(field) for m in metadata_dicts]
            if len(set(values)) > 1:
                logger.warning(f"Different {field} values across archives: {values}")
    
    def _read_header_from_file(self, file) -> tuple[dict, int]:
        """Read header from a specific file."""
        
        magic = file.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid file format. Expected {MAGIC_BYTES}, got {magic}")
        
        version = struct.unpack('<I', file.read(4))[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported version {version}. Expected {FORMAT_VERSION}")
        
        
        record_count = struct.unpack('<I', file.read(4))[0]
        
        metadata_length = struct.unpack('<I', file.read(4))[0]
        metadata_json = file.read(metadata_length).decode('utf-8')
        metadata_dict = json.loads(metadata_json)
        
        file.seek(HEADER_SIZE)
        
        return metadata_dict, record_count
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close the file(s)."""
        for file in self.files:
            if file:
                file.close()
        self.files = []
            
    def __len__(self):
        """Get number of records."""
        return self.record_count
    
    def __iter__(self) -> Iterator['ReadResult']:
        yield from self.reads()

    def reads(self,
              selection: Optional[Iterable[str]] = None
              ) -> Generator["ReadResult", None, None]:
        """Iterate through all records sequentially."""
        for file_idx, (file, record_count) in enumerate(zip(self.files, self.record_counts)):
            self._current_file_idx = file_idx
            file.seek(HEADER_SIZE)
            
            for _ in range(record_count):
                next_read = self._read_next_record_from_file(file_idx)
                next_read_id = next_read.read_id
                if selection and next_read_id not in selection:
                    continue
                yield next_read

    def _read_next_record_from_file(self, file_idx: int) -> 'ReadResult':
        """Read the next record from specific file."""
        file = self.files[file_idx]
        
        # Read and verify record marke
        marker = file.read(len(RECORD_MARKER))
        if marker != RECORD_MARKER:
            raise ValueError(f"Invalid record marker at position {file.tell()}")
        
        # Read sizes
        compressed_size = struct.unpack('<I', file.read(4))[0]
        uncompressed_size = struct.unpack('<I', file.read(4))[0]
        
        # Read and decompress data
        compressed_data = file.read(compressed_size)
        data = self.decompressors[file_idx].decompress(compressed_data)
        
        if len(data) != uncompressed_size:
            raise ValueError(f"Decompression size mismatch: expected {uncompressed_size}, got {len(data)}")
        
        # Parse the data
        offset = 0
        
        # Read ID
        read_id_len = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        read_id = data[offset:offset + read_id_len].decode('utf-8')
        offset += read_id_len
        
        # Read metadata
        num_chunks = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        chunk_size = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        
        # Read arrays
        logits = {}
        
        # Classification logits
        has_classification = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        if has_classification:
            array_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            cls_bytes = data[offset:offset + array_len * 4]  # float32 = 4 bytes
            logits['seq_class'] = np.frombuffer(cls_bytes, dtype='float32')
            offset += array_len * 4
        
        # Seq2seq logits
        has_seq2seq = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        if has_seq2seq:
            shape_0, shape_1 = struct.unpack_from('<II', data, offset)
            offset += 8
            seq_bytes = data[offset:offset + shape_0 * shape_1 * 4]
            logits['seq2seq'] = np.frombuffer(seq_bytes, dtype='float32').reshape(shape_0, shape_1)
        
        # Import ReadResult only when needed
        from ...storages import ReadResult
        
        return ReadResult(
            read_id=read_id,
            _logits=logits,
            num_chunks=num_chunks,
            chunk_size=chunk_size
        )
    
    def build_index(self):
        """Build index for random access. Scans entire file once."""
        if self._index is not None:
            logger.info("Index already built")
            return
        
        self._index = {}  # read_id -> (file_idx, offset, size)
        
        for file_idx, (file, record_count) in enumerate(
            zip(self.files, self.record_counts)
        ):
            file.seek(HEADER_SIZE)
            
            for i in range(record_count):
                # Remember position before record
                pos = file.tell()
                
                # Read record marker
                marker = file.read(len(RECORD_MARKER))
                if marker != RECORD_MARKER:
                    raise ValueError(f"Invalid record marker at position {pos}")
                
                # Read sizes
                compressed_size = struct.unpack('<I', file.read(4))[0]
                uncompressed_size = struct.unpack('<I', file.read(4))[0]
                
                # Read compressed data to get read_id
                compressed_data = file.read(compressed_size)
                
                # We need to decompress just enough to get the read_id
                data = self.decompressors[file_idx].decompress(compressed_data)
                
                # Extract read_id
                read_id_len = struct.unpack_from('<H', data, 0)[0]
                read_id = data[2:2 + read_id_len].decode('utf-8')
                
                # Store position and size in index with file index
                self._index[read_id] = {
                    'file_idx': file_idx,
                    'offset': pos,
                    'record_size': len(RECORD_MARKER) + 4 + 4 + compressed_size
                }
    
    def get_read(self, read_id: str) -> 'ReadResult':
        """Get specific result by read_id. Builds index on first use."""
        if self._index is None:
            self.build_index()
        
        if read_id not in self._index:
            raise KeyError(f"Read ID '{read_id}' not found in archive")
        
        # Get file and position
        record_info = self._index[read_id]
        file_idx = record_info['file_idx']
        file = self.files[file_idx]
        
        # Seek to record position
        file.seek(record_info['offset'])
        
        # Read the record
        return self._read_next_record_from_file(file_idx)
    
    def get_path(self, read_id: str) -> Path:
        """Get the path of the archive containing this read_id."""
        if self._index is None:
            self.build_index()
        
        if read_id not in self._index:
            raise KeyError(f"Read ID '{read_id}' not found in any archive")
        
        file_idx = self._index[read_id]['file_idx']
        return self._paths[file_idx]
    
    def __contains__(self, read_id: str) -> bool:
        """Check if read_id exists in archive."""
        if self._index is None:
            self.build_index()
        return read_id in self._index
    
    @property
    def read_ids(self) -> list:
        """Get all read IDs. Builds index if needed."""
        if self._index is None:
            self.build_index()
        return list(self._index.keys())
    
    @property
    def metadata(self) -> "InferenceMetadata":
        try:
            from ...storages import InferenceMetadata
            metadata_dict = self.metadata_dict.copy()
            metadata = InferenceMetadata(**metadata_dict)
            return metadata
        except:
            raise ValueError("Cannot reconstruct metadata")
    
    def to_inference_results(self) -> 'InferenceResults':
        """Convert entire archive to InferenceResults object."""
        from ...storages import InferenceResults, InferenceMetadata
        
        # Reconstruct metadata
        metadata_dict = self.metadata_dict.copy()
        if metadata_dict.get('pod5_paths') is not None:
            metadata_dict['pod5_paths'] = set(metadata_dict['pod5_paths'])
            
        metadata = InferenceMetadata(**metadata_dict)
        results = InferenceResults(metadata=metadata)
        
        # Add all results
        for result in self:
            results._add_result(result)
            
        return results
    
    def summary(self) -> dict:
        """Get archive summary information."""
        if self.is_multi:
            return {
                'num_archives': len(self._paths),
                'paths': [str(p) for p in self._paths],
                'total_record_count': self.record_count,
                'record_counts_per_file': self.record_counts,
                'indexed': self._index is not None,
                'index_size': len(self._index) if self._index else 0
            }
        else:
            return {
                'path': str(self._paths[0]),
                'format_version': FORMAT_VERSION,
                'record_count': self.record_count,
                'model_name': self.metadata_dict.get('model_name'),
                'chunk_size': self.metadata_dict.get('chunk_size'),
                'indexed': self._index is not None,
                'index_size': len(self._index) if self._index else 0
            }