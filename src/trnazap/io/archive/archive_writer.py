import struct
import json
import numpy as np
import zstandard as zstd
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union
from dataclasses import asdict
import logging
import uuid

from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    COMPRESSION_LEVEL, RECORD_MARKER,
    INDEX_MAGIC, FOOTER_MAGIC, ENC_UUID16, ENC_UTF8LEN
)

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, ReadResult

logger = logging.getLogger(__name__)


class ZIRWriter:
    """Write inference results to ZIR archive format with dynamic logit support."""
    
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
        self._index_entries = []  # list[tuple[id_bytes: bytes, offset: int, size: int]]
        self._index_encoding = None
        
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
                
                # write footer index
                footer_start = self.file.tell()
                self._write_footer_index(footer_start)
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
        try:
            metadata_dict = asdict(self.metadata)
        except TypeError:
            metadata_dict = dict(self.metadata)


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
        

    def _write_footer_index(self, footer_start: int) -> None:
        enc = self._index_encoding or ENC_UTF8LEN
        self.file.write(INDEX_MAGIC)
        self.file.write(struct.pack('<H', enc))
        self.file.write(struct.pack('<I', len(self._index_entries)))

        if enc == ENC_UUID16:
            for id_bytes, offset, size in self._index_entries:
                # id_bytes must be 16
                if len(id_bytes) != 16:
                    raise ValueError("UUID index expects 16-byte IDs")
                self.file.write(id_bytes)
                self.file.write(struct.pack('<Q', offset))
                self.file.write(struct.pack('<I', size))
        else:  # ENC_UTF8LEN
            for id_bytes, offset, size in self._index_entries:
                self.file.write(struct.pack('<H', len(id_bytes)))
                self.file.write(id_bytes)
                self.file.write(struct.pack('<Q', offset))
                self.file.write(struct.pack('<I', size))

        self.file.write(FOOTER_MAGIC)
        self.file.write(struct.pack('<Q', footer_start))
        
    def add_result(self, read_result: 'ReadResult'):
        """
        Add a single read_result to the archive.
        
        Args:
            read_result: ReadResult to add
        """
        if not self.file:
            raise RuntimeError("Writer not opened. Use 'with' statement.")
        
        # Capture position before writing this record
        record_start = self.file.tell()
    
        # Prepare data for compression
        buffer = self._serialize_result(read_result)
        
        # Compress the buffer
        compressed = self.compressor.compress(buffer)
        
        # Write frame
        self.file.write(RECORD_MARKER)
        self.file.write(struct.pack('<I', len(compressed)))
        self.file.write(struct.pack('<I', len(buffer)))
        self.file.write(compressed)
        
        # Compute framed size
        framed_size = len(RECORD_MARKER) + 4 + 4 + len(compressed)
        rid = read_result.read_id
        id_tag = None
        enc = None
        try:
            # Try UUID fast-path
            # Accept both 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' and uppercase
            u = uuid.UUID(rid)
            id_tag = u.bytes   # 16 bytes
            enc = ENC_UUID16
        except Exception:
            rb = rid.encode('utf-8')
            if len(rb) > 0xFFFF:
                raise ValueError("read_id too long for uint16 length")
            id_tag = rb
            enc = ENC_UTF8LEN

        # Enforce single encoding per file (first record decides)
        if self._index_encoding is None:
            self._index_encoding = enc
        elif self._index_encoding != enc:
            # If mixed, fall back to UTF8_LEN (rare; or you can raise)
            if self._index_encoding == ENC_UUID16 and enc == ENC_UTF8LEN:
                # convert previous UUID entries to UTF8 if needed — simpler: just raise
                raise ValueError("Mixed read_id formats detected; prefer consistent UUIDs.")
            else:
                raise ValueError("Mixed read_id formats detected.")

        self._index_entries.append((id_tag, record_start, framed_size))
        self.record_count += 1

    def _serialize_result(self, read_result):
        """
        Serialize result into a byte buffer (before compression).
        Now supports dynamic logit keys.
        
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
        
        # NEW: Write number of logit entries
        logit_entries = read_result._logits
        buffer.extend(struct.pack('<B', len(logit_entries)))  # 1 byte for count (max 255 tasks)
        
        # Write each logit entry dynamically
        for key, logits in logit_entries.items():
            # Write key name
            key_bytes = key.encode('utf-8')
            buffer.extend(struct.pack('<B', len(key_bytes)))  # 1 byte for key length
            buffer.extend(key_bytes)
            
            # Write array info
            array = logits.astype('float32')
            buffer.extend(struct.pack('<B', array.ndim))  # 1 byte for number of dimensions
            
            # Write shape
            for dim in array.shape:
                buffer.extend(struct.pack('<I', dim))  # 4 bytes per dimension
            
            # Write data
            buffer.extend(array.tobytes())
        
        return bytes(buffer)
        
        
class ZIRShardManager:
    """Manages writing to ZIR files with optional sharding and comprehensive path handling."""
    
    def __init__(
        self,
        base_path: Union[str, Path],
        metadata: 'InferenceMetadata',
        shard_size: Optional[int] = None,
    ):
        """
        Initialize shard manager with extensive path management.
        
        Args:
            base_path: Base path for output file(s)
                - If shard_size is None: Must be a file path ending with .zir
                - If shard_size is set:
                    - Directory: Files saved as dir/shard0000.zir, dir/shard0001.zir
                    - Path without extension: /path/save → /path/save_shard0000.zir
                    - Path with extension: Extension ignored, user warned
            metadata: Inference metadata
            shard_size: Records per shard (None for single file)
            
        Raises:
            ValueError: If path handling fails
        """
        self.metadata = metadata
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_writer: Optional[ZIRWriter] = None
        self.current_count = 0
        self.total_reads = 0
        
        # Process and validate base path
        self._process_base_path(base_path)
        
        # Create necessary directories
        self._ensure_directories()
        
        # Open first shard/file
        self._open_new_shard()
    
    def _process_base_path(self, base_path: Union[str, Path]) -> None:
        """Process and validate the base path according to sharding mode."""
        input_path = Path(base_path)
        
        if self.shard_size is None:
            if input_path.suffix.lower() != '.zir':
                if input_path.suffix:
                    self.base_path = input_path.with_suffix('.zir')
                    logger.warning(
                        f"Single file mode: Changed extension from '{input_path.suffix}' to '.zir'. "
                        f"Output will be: {self.base_path}"
                    )
                else:
                    self.base_path = input_path.with_suffix('.zir')
                    logger.warning(f"Single file mode: Added .zir extension. Output will be: {self.base_path}")
            else:
                self.base_path = input_path
                
            self.shard_pattern = None
            self.output_dir = self.base_path.parent
            
        else:
            if input_path.suffix:
                self.output_dir = input_path.parent
                base_name = input_path.stem
                self.shard_pattern = f"{base_name}_shard{{:04d}}.zir"
                logger.warning(
                    f"Sharding mode: Extension '{input_path.suffix}' ignored. "
                    f"Files will be saved as: {self.output_dir}/{base_name}_shard0000.zir, etc."
                )
                
            else:
                self.output_dir = input_path
                self.shard_pattern = "shard{:04d}.zir"
            
            # Store the pattern for reference
            self.base_path = input_path  # Keep original for reference
    
    def _ensure_directories(self) -> None:
        """Create necessary directories."""
        if self.shard_size is None:
            # Single file mode - create parent directory if needed
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Sharding mode - create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, read_result: 'ReadResult') -> None:
        """Add a result, potentially opening a new shard."""
        if self.shard_size and self.current_count >= self.shard_size:
            self._close_current_shard()
            self._open_new_shard()
        
        self.current_writer.add_result(read_result)
        self.current_count += 1
        self.total_reads += 1
    
    def _get_shard_path(self) -> Path:
        """Generate shard filename based on mode."""
        if self.shard_size is None:
            # Single file mode
            return self.base_path
        else:
            # Sharding mode
            filename = self.shard_pattern.format(self.current_shard)
            return self.output_dir / filename
    
    def _open_new_shard(self) -> None:
        """Open a new shard file."""
        shard_path = self._get_shard_path()
        self.current_writer = ZIRWriter(shard_path, self.metadata)
        self.current_writer.__enter__()
        self.current_count = 0
        
    
    def _close_current_shard(self) -> None:
        """Close current shard."""
        if self.current_writer:
            self.current_writer.__exit__(None, None, None)
            
            self.current_writer = None
            self.current_shard += 1
    
    def close(self) -> None:
        """Close any open shard and finalize."""
        self._close_current_shard()