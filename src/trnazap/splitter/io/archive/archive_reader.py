# src/trnazap/splitter/io/archive/archive_reader.py
"""Reader for ZIR archive format."""

import struct
import json
import zstandard as zstd
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    RECORD_MARKER
)

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, ReadResult, InferenceResults

logger = logging.getLogger(__name__)


class ZIRReader:
    """Read inference results from ZIR archive format."""
    
    def __init__(self, path: Path, index=False):
        """
        Initialize reader.
        
        Args:
            path: Archive file path
        """
        self.path = Path(path)
        self.file = open(self.path, 'rb')
        self.decompressor = zstd.ZstdDecompressor()
        
        # Read and validate header
        self._read_header()
        
        # Index will be built on demand
        self._index = None
        self._current_pos = HEADER_SIZE  # Position after header

        # Build index if demanded
        if index:
            self.build_index()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()
            self.file = None
            
    def _read_header(self):
        """Read and validate file header."""
        # Check magic bytes
        magic = self.file.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid file format. Expected {MAGIC_BYTES}, got {magic}")
        
        # Check version
        version = struct.unpack('<I', self.file.read(4))[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported version {version}. Expected {FORMAT_VERSION}")
        
        # Read record count
        self.record_count = struct.unpack('<I', self.file.read(4))[0]
        
        # Read metadata
        metadata_length = struct.unpack('<I', self.file.read(4))[0]
        metadata_json = self.file.read(metadata_length).decode('utf-8')
        self.metadata_dict = json.loads(metadata_json)
        
        # Skip to end of header
        self.file.seek(HEADER_SIZE)
    
    def __len__(self):
        """Get number of records."""
        return self.record_count
    
    # =============================================================================
    # InferenceResults Container
    # =============================================================================
    
    def __iter__(self) -> Iterator['ReadResult']:
        """Iterate through all records sequentially."""
        self.file.seek(HEADER_SIZE)  # Reset to start of data
        
        for _ in range(self.record_count):
            yield self._read_next_record()

    def _read_next_record(self) -> 'ReadResult':
        """Read the next record from current file position."""
        # Read and verify record marker
        marker = self.file.read(len(RECORD_MARKER))
        if marker != RECORD_MARKER:
            raise ValueError(f"Invalid record marker at position {self.file.tell()}")
        
        # Read sizes
        compressed_size = struct.unpack('<I', self.file.read(4))[0]
        uncompressed_size = struct.unpack('<I', self.file.read(4))[0]
        
        # Read and decompress data
        compressed_data = self.file.read(compressed_size)
        data = self.decompressor.decompress(compressed_data)
        
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
    
    # =============================================================================
    # Indexed accessed
    # =============================================================================
    
    def build_index(self):
        """Build index for random access. Scans entire file once."""
        if self._index is not None:
            logger.info("Index already built")
            return
        
        self._index = {}
        self.file.seek(HEADER_SIZE)
        
        for i in range(self.record_count):
            # Remember position before record
            pos = self.file.tell()
            
            # Read record marker
            marker = self.file.read(len(RECORD_MARKER))
            if marker != RECORD_MARKER:
                raise ValueError(f"Invalid record marker at position {pos}")
            
            # Read sizes
            compressed_size = struct.unpack('<I', self.file.read(4))[0]
            uncompressed_size = struct.unpack('<I', self.file.read(4))[0]
            
            # Read compressed data to get read_id
            compressed_data = self.file.read(compressed_size)
            
            # We need to decompress just enough to get the read_id
            data = self.decompressor.decompress(compressed_data)
            
            # Extract read_id
            read_id_len = struct.unpack_from('<H', data, 0)[0]
            read_id = data[2:2 + read_id_len].decode('utf-8')
            
            # Store position and size in index
            self._index[read_id] = {
                'offset': pos,
                'record_size': len(RECORD_MARKER) + 4 + 4 + compressed_size
            }
    
    def get_read(self, read_id: str) -> 'ReadResult':
        """Get specific result by read_id. Builds index on first use."""
        if self._index is None:
            self.build_index()
        
        if read_id not in self._index:
            raise KeyError(f"Read ID '{read_id}' not found in archive")
        
        # Seek to record position
        record_info = self._index[read_id]
        self.file.seek(record_info['offset'])
        
        # Read the record
        return self._read_next_record()
    
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
        return {
            'path': str(self.path),
            'format_version': FORMAT_VERSION,
            'record_count': self.record_count,
            'model_name': self.metadata_dict.get('model_name'),
            'chunk_size': self.metadata_dict.get('chunk_size'),
            'indexed': self._index is not None,
            'index_size': len(self._index) if self._index else 0
        }