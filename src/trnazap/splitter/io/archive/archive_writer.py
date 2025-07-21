import struct
import json
import numpy as np
import zstandard as zstd
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import asdict
import logging

from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    COMPRESSION_ALGO, COMPRESSION_LEVEL, RECORD_MARKER
)

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, ReadResult

logger = logging.getLogger(__name__)


class ZIRWriter:
    """Write inference results to ZIR archive format."""
    
    def __init__(self, path: Path, metadata: 'InferenceMetadata'):
        """
        Initialize writer.
        
        Args:
            path: Output file path
            metadata: Inference metadata
        """
        self.path = Path(path)
        self.metadata = metadata
        self.file = None
        self.compressor = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)
        self.record_count = 0
        
    def __enter__(self):
        """Open file and write header."""
        self.file = open(self.path, 'wb')
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file and update record count."""
        if self.file:
            try:
                current_pos = self.file.tell()
                self.file.seek(len(MAGIC_BYTES) + 4)  # After magic + version
                self.file.write(struct.pack('<I', self.record_count))
                self.file.seek(current_pos)
            finally:
                self.file.close()
                self.file = None
            
    def _write_header(self) -> None:
        """Write archive header with metadata and padding."""
        # Write magic bytes and version
        self.file.write(MAGIC_BYTES)
        self.file.write(struct.pack('<I', FORMAT_VERSION))

        # Write placeholder for record count (will update on close)
        self.file.write(struct.pack('<I', 0))

        # Prepare metadata
        metadata_dict = asdict(self.metadata)

        # Handle special types
        if metadata_dict.get('pod5_paths') is not None:
            metadata_dict['pod5_paths'] = list(metadata_dict['pod5_paths'])

        metadata_json = json.dumps(metadata_dict, default=str)
        metadata_bytes = metadata_json.encode('utf-8')
        metadata_len = len(metadata_bytes)

        # Write metadata length and data
        self.file.write(struct.pack('<I', metadata_len))
        self.file.write(metadata_bytes)

        # Check for padding
        total_written = len(MAGIC_BYTES) + 4 + 4 + 4 + metadata_len
        padding = HEADER_SIZE - total_written
        if padding < 0:
            raise ValueError(f"Metadata too large to fit in header: {metadata_len} bytes")

        # Pad to HEADER_SIZE
        self.file.write(b'\x00' * padding)
        
    def add_result(self, read_result: 'ReadResult'):
        """
        Add a single read_result to the archive.
        
        Args:
            read_result: ReadResult to add
        """
        if not self.file:
            raise RuntimeError("Writer not opened. Use 'with' statement.")
        
        # Prepare data for compression
        buffer = self._serialize_result(read_result)
        
        # Compress the buffer
        compressed = self.compressor.compress(buffer)
        
        # Write record marker
        self.file.write(RECORD_MARKER)
        
        # Write compressed and uncompressed sizes
        self.file.write(struct.pack('<I', len(compressed)))
        self.file.write(struct.pack('<I', len(buffer)))
        
        # Write compressed data
        self.file.write(compressed)
        
        self.record_count += 1

    def _serialize_result(self, read_result):
        """
        Serialize result into a byte buffer (before compression).
        Returns:
            bytes: uncompressed binary data for a single record
        """
        buffer = bytearray()
        
        # Write read ID
        read_id_bytes = read_result.read_id.encode('utf-8')
        buffer.extend(struct.pack('<H', len(read_id_bytes)))  # 2 bytes for length
        buffer.extend(read_id_bytes)
        
        # Write basic metadata
        buffer.extend(struct.pack('<i', read_result.num_chunks))  # 4 bytes
        buffer.extend(struct.pack('<i', read_result.chunk_size))  # 4 bytes
        
        # Write classification logits if present
        if read_result.classification_logits is not None:
            buffer.extend(struct.pack('<B', 1))  # Has classification flag
            cls_array = read_result.classification_logits.astype('float32')
            buffer.extend(struct.pack('<I', len(cls_array)))  # Array length
            buffer.extend(cls_array.tobytes())
        else:
            buffer.extend(struct.pack('<B', 0))  # No classification flag
        
        # Write seq2seq logits if present
        if read_result.seq2seq_logits is not None:
            buffer.extend(struct.pack('<B', 1))  # Has seq2seq flag
            seq_array = read_result.seq2seq_logits.astype('float32')
            shape = seq_array.shape
            buffer.extend(struct.pack('<II', shape[0], shape[1]))  # Shape (T, 4)
            buffer.extend(seq_array.tobytes())
        else:
            buffer.extend(struct.pack('<B', 0))  # No seq2seq flag

        return bytes(buffer)