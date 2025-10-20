import struct
import json
import os
import io
import numpy as np
import zstandard as zstd
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union
from dataclasses import asdict
import logging
import uuid

from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    COMPRESSION_LEVEL, RECORD_MARKER, BUFFER_SIZE
)

if TYPE_CHECKING:
    from ..storages import InferenceMetadata, ReadResult, ReadResultCompressed

logger = logging.getLogger(__name__)


class ZIRWriter:
    """High-performance ZIR archive writer with dynamic logits.

    IO-optimized design notes:
    - Uses a large buffered writer to minimize syscalls.
    - Caches struct packers to reduce overhead.
    - Streams zstd frames per record with checksums for integrity.
    - Performs strict bounds checks (<=255 logits/keys, <=255 dims).
    - Avoids unnecessary copies when possible (e.g., astype(copy=False)).

    Record layout (per record):
        RECORD_MARKER
        compressed_size : uint32 LE
        uncompressed_size : uint32 LE
        compressed_payload : bytes[compressed_size]

    Uncompressed payload layout:
        read_id_len : uint16 LE
        read_id     : utf8 bytes[read_id_len]
        num_chunks  : int32 LE
        chunk_size  : int32 LE
        num_logits  : uint8
        repeat num_logits times:
            key_len : uint8
            key     : utf8 bytes[key_len]
            ndim    : uint8
            shape   : uint32 LE [ndim]
            data    : float32 little-endian bytes[prod(shape)*4]
    """

    __slots__ = (
        "path",
        "metadata",
        "_file",
        "_buf",
        "_zstd",
        "record_count",
        "_S_U16",
        "_S_U32",
        "_S_I32",
    )

    def __init__(
        self,
        path: Path,
        metadata: "InferenceMetadata",
        *,
        compression_level: int = COMPRESSION_LEVEL,
        buffered_bytes: int = BUFFER_SIZE,
        write_checksum: bool = True,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self._file: Optional[io.BufferedWriter] = None
        self._buf = buffered_bytes
        self._zstd = zstd.ZstdCompressor(level=compression_level, 
                                         write_checksum=write_checksum, 
                                         write_content_size=True)
        self.record_count = 0

        # Cached struct packers
        self._S_U16 = struct.Struct("<H")
        self._S_U32 = struct.Struct("<I")
        self._S_I32 = struct.Struct("<i")

    # ---------------- Context Manager ---------------- #
    def __enter__(self) -> "ZIRWriter":
        # Open with explicit buffering; binary exclusive create/truncate
        raw = open(self.path, "wb", buffering=0)
        self._file = io.BufferedWriter(raw, buffer_size=self._buf)
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._file:
            return
        try:
            # Seek and patch record_count (magic + version)
            self._file.flush()
            self._file.raw.seek(len(MAGIC_BYTES) + 4)
            self._file.raw.write(self._S_U32.pack(self.record_count))
            # Return to end to ensure consistent file position
            self._file.raw.seek(0, os.SEEK_END)
            self._file.flush()
            try:
                os.fsync(self._file.raw.fileno())
            except Exception:
                # fsync may be unavailable on some platforms; ignore
                pass
        finally:
            self._file.close()
            self._file = None

    # ---------------- Public API ---------------- #
    def add_result(self, read_result: Union["ReadResult", "ReadResultCompressed"]) -> None:
        """Append a single read result as a compressed record.

        Args:
            read_result: Object with fields
                - read_id (str)
                - num_chunks (int)
                - chunk_size (int)
                - _logits (Mapping[str, np.ndarray-like])
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use 'with ZIRWriter(...) as w:'")

        payload = self._serialize_result(read_result)

        compressed = self._zstd.compress(payload)

        # Write record marker + sizes + frame
        f = self._file
        f.write(RECORD_MARKER)
        f.write(self._S_U32.pack(len(compressed)))
        f.write(self._S_U32.pack(len(payload)))
        f.write(compressed)

        self.record_count += 1

    # ---------------- Internal: Header ---------------- #
    def _write_header(self) -> None:
        assert self._file is not None
        f = self._file

        # Magic + version
        f.write(MAGIC_BYTES)
        f.write(self._S_U32.pack(FORMAT_VERSION))

        # Placeholder for record_count
        f.write(self._S_U32.pack(0))

        # Metadata JSON (convert sets to lists, Path to str, etc.)
        meta = asdict(self.metadata)
        if meta.get("pod5_paths") is not None and not isinstance(meta["pod5_paths"], list):
            meta["pod5_paths"] = list(meta["pod5_paths"])  # ensure JSON-serializable

        meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
        meta_bytes = meta_json.encode("utf-8")
        meta_len = len(meta_bytes)

        # Header layout reserves HEADER_SIZE; enforce fit
        fixed = len(MAGIC_BYTES) + 4 + 4 + 4  # magic + version + record_count + meta_len
        total = fixed + meta_len
        padding = HEADER_SIZE - total
        if padding < 0:
            raise ValueError(
                f"Metadata too large for fixed header: {meta_len} bytes (max {HEADER_SIZE - fixed})"
            )

        f.write(self._S_U32.pack(meta_len))
        f.write(meta_bytes)
        if padding:
            f.write(b"\x00" * padding)

    # ---------------- Internal: Serialization ---------------- #
    def _serialize_result(self, rr: Union["ReadResult", "ReadResultCompressed"]) -> bytes:
        # Validate and prepare fields common to both
        read_id = rr.read_id
        if not isinstance(read_id, str):
            raise TypeError("read_id must be str")
        read_id_b = read_id.encode("utf-8")
        if len(read_id_b) > 0xFFFF:
            raise ValueError("read_id is too long (>65535 bytes)")

        num_chunks = int(rr.num_chunks)
        chunk_size = int(rr.chunk_size)

        # Detect summary vs full: full has _logits; summary has top3_classes and no _logits
        has_logits = hasattr(rr, "_logits") and getattr(rr, "_logits") is not None
        is_summary = not has_logits  # i.e., ReadResultCompressed

        buf = io.BytesIO()

        # read_id
        buf.write(self._S_U16.pack(len(read_id_b)))
        buf.write(read_id_b)

        # basic metadata
        buf.write(self._S_I32.pack(num_chunks))
        buf.write(self._S_I32.pack(chunk_size))

        # record_kind
        buf.write(struct.pack("<B", 1 if is_summary else 0))

        if is_summary:
            # ---- SUMMARY PAYLOAD (JSON blob) ----
            # Build a compact JSON; ensure numpy types are converted
            summary = {
                "top3_classes": (rr.top3_classes.tolist()
                                if hasattr(rr, "top3_classes") and isinstance(rr.top3_classes, np.ndarray)
                                else list(rr.top3_classes) if hasattr(rr, "top3_classes") else []),
                "variable_region_range": getattr(rr, "variable_region_range", (-1, -1)),
                "smoothed_variable_region_range": getattr(rr, "smoothed_variable_region_range", (-1, -1)),
                "fragmented": bool(getattr(rr, "fragmented", False)),
            }
            s = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            buf.write(self._S_U32.pack(len(s)))
            buf.write(s)
            return buf.getvalue()

        # ---- FULL PAYLOAD (logits blocks) ----
        logits = getattr(rr, "_logits", {})  # Mapping[str, np.ndarray-like]
        if not isinstance(logits, dict):
            logits = dict(logits)

        n_entries = len(logits)
        if n_entries > 255:
            raise ValueError("At most 255 logit entries are supported")

        # number of logits
        buf.write(struct.pack("<B", n_entries))

        # (Optionally) make ordering deterministic across runs:
        for key in sorted(logits.keys()):
            arr = logits[key]

            if not isinstance(key, str):
                raise TypeError("logit key must be str")
            key_b = key.encode("utf-8")
            if len(key_b) > 255:
                raise ValueError(f"logit key '{key}' too long (>255 bytes)")

            # Convert to float32 little-endian without unnecessary copies
            a = np.asarray(arr)
            if a.dtype != np.dtype("<f4"):
                a = a.astype("<f4", copy=False)  # explicit little-endian float32

            ndim = int(a.ndim)
            if not (1 <= ndim <= 255):
                raise ValueError(f"ndim must be in [1, 255], got {ndim}")

            shape = tuple(int(d) for d in a.shape)
            for d in shape:
                if d < 0:
                    raise ValueError("negative dimensions are not allowed")
                if d > 0xFFFFFFFF:
                    raise ValueError("dimension too large for uint32 shape")

            # key
            buf.write(struct.pack("<B", len(key_b)))
            buf.write(key_b)

            # ndim + shape
            buf.write(struct.pack("<B", ndim))
            if ndim == 1:
                buf.write(self._S_U32.pack(shape[0]))
            else:
                buf.write(struct.pack("<" + "I" * ndim, *shape))

            # data (raw bytes, little-endian float32)
            # Ensure C-contiguity WITHOUT changing dtype/endian:
            if not a.flags.c_contiguous:
                a = np.ascontiguousarray(a)
            mv = memoryview(a)
            buf.write(mv.cast("b"))

        return buf.getvalue()


        
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