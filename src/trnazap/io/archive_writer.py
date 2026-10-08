from __future__ import annotations
import struct
import json
import os
import io
import numpy as np
import zstandard as zstd
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING, Union
from dataclasses import asdict
import logging
import time
import uuid

from .archive_format import (
    MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE,
    COMPRESSION_LEVEL, RECORD_MARKER, BUFFER_SIZE
)

if TYPE_CHECKING:
    from ..storages import InferenceMetadata, ReadResultLike, InferenceStore

logger = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial"
TMP_DIR_PREFIX = ".tmp_"


def _optional_bool(v):
    return None if v is None else bool(v)


def new_run_id() -> str:
    """A new run id, ``<timestamp>_<6 hex>``."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def run_id_of_tmp_dir(tmp_dir: Path) -> str:
    """The run id a temp dir is named after (``.tmp_<run_id>``)."""
    return Path(tmp_dir).name[len(TMP_DIR_PREFIX):]


def make_tmp_dir(output_dir: Path, run_id: Optional[str] = None) -> Path:
    """Create ``<output_dir>/.tmp_<run_id>/`` for one run (a new id if None).

    If that name is taken (e.g. a failed run's dir being re-sharded into the same
    place), a short random suffix is added rather than reusing it.
    """
    tmp_dir = Path(output_dir) / f"{TMP_DIR_PREFIX}{run_id or new_run_id()}"
    if tmp_dir.exists():
        tmp_dir = tmp_dir.with_name(f"{tmp_dir.name}_{uuid.uuid4().hex[:4]}")
    tmp_dir.mkdir()
    return tmp_dir


def finalize_tmp_dir(tmp_dir: Path, output_dir: Path) -> List[Path]:
    """Move every ``<name>.zir.partial`` in ``tmp_dir`` to ``output_dir/<name>.zir``.

    Then remove ``tmp_dir`` with rmdir (not rmtree), so anything unexpected left in
    it raises instead of being deleted. Returns the final paths.
    """
    finals = []
    for partial in sorted(Path(tmp_dir).glob(f"*.zir{PARTIAL_SUFFIX}")):
        final = Path(output_dir) / partial.name[: -len(PARTIAL_SUFFIX)]
        os.replace(partial, final)
        finals.append(final)
    Path(tmp_dir).rmdir()
    return finals


def _header_bytes(metadata: "InferenceMetadata", record_count: int) -> bytes:
    """The fixed-size header: magic, version, record count, metadata JSON, padding."""
    meta = asdict(metadata)
    if meta.get("pod5_paths") is not None and not isinstance(meta["pod5_paths"], list):
        meta["pod5_paths"] = list(meta["pod5_paths"])  # ensure JSON-serializable
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    fixed = len(MAGIC_BYTES) + 4 + 4 + 4  # magic + version + record_count + meta_len
    padding = HEADER_SIZE - fixed - len(meta_bytes)
    if padding < 0:
        raise ValueError(
            f"Metadata too large for fixed header: {len(meta_bytes)} bytes "
            f"(max {HEADER_SIZE - fixed})"
        )
    S_U32 = struct.Struct("<I")
    return b"".join((
        MAGIC_BYTES,
        S_U32.pack(FORMAT_VERSION),
        S_U32.pack(record_count),
        S_U32.pack(len(meta_bytes)),
        meta_bytes,
        b"\x00" * padding,
    ))


def rewrite_header(path: Path, metadata: "InferenceMetadata") -> None:
    """Replace a closed archive's header metadata, keeping its record count.

    Used at the end of a run so every shard carries the final run totals. If the
    new metadata doesn't fit the fixed header, the old header is kept (warning).
    """
    with open(path, "r+b") as f:
        f.seek(len(MAGIC_BYTES) + 4)
        (record_count,) = struct.unpack("<I", f.read(4))
        try:
            header = _header_bytes(metadata, record_count)
        except ValueError as exc:
            logger.warning("Kept original header of %s: %s", path, exc)
            return
        f.seek(0)
        f.write(header)


def warn_existing_outputs(output_dir: Path, existing: List[Path]) -> None:
    """Warn (never block) about archives a run will overwrite and leftover temp dirs."""
    if existing:
        logger.warning(
            "Output already contains %d archive(s) matching this run's names (e.g. %s); "
            "they will be overwritten, and any not rewritten will mix with this run's results",
            len(existing), existing[0],
        )
    stale = sorted(Path(output_dir).glob(f"{TMP_DIR_PREFIX}*"))
    if stale:
        logger.warning(
            "Output contains %d temp dir(s) left by unfinished runs (e.g. %s)",
            len(stale), stale[0],
        )


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
        compressed_size   : uint32 LE
        uncompressed_size : uint32 LE
        compressed_payload : bytes[compressed_size]

    Uncompressed payload layout (common prefix):
        read_id_len : uint16 LE
        read_id     : utf8 bytes[read_id_len]
        num_chunks  : int32 LE
        chunk_size  : int32 LE
        record_kind : uint8        # 0 = FULL (logits), 1 = SUMMARY (compressed)

    record_kind == 0  (FULL — ReadResultDetailed):
        num_logits : uint8
        repeat num_logits times:
            key_len : uint8
            key     : utf8 bytes[key_len]
            ndim    : uint8
            shape   : uint32 LE [ndim]
            data    : float32 little-endian bytes[prod(shape) * 4]

    record_kind == 1  (SUMMARY — ReadResult):
        summary_len : uint32 LE
        summary     : utf8 JSON bytes[summary_len]
                      keys: top3_classes, variable_region_range,
                            smoothed_variable_region_range, fragmented
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
            # Rewrite the header with the record count and the metadata as it is
            # now (end-of-run fields); if it no longer fits, patch only the count.
            self._file.flush()
            try:
                header = _header_bytes(self.metadata, self.record_count)
                self._file.raw.seek(0)
                self._file.raw.write(header)
            except ValueError as exc:
                logger.warning("Kept original header metadata of %s: %s", self.path, exc)
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
    def add_result(self, read_result: ReadResultLike) -> None:
        """Append a single read result as a compressed record.

        Args:
            read_result: ReadResultDetailed Object with fields
                - read_id (str)
                - num_chunks (int)
                - chunk_size (int)
                - _logits (Mapping[str, np.ndarray-like])
                
            OR
            
            read_result: ReadResult Object
            
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
        self._file.write(_header_bytes(self.metadata, 0))  # count patched at close

    # ---------------- Internal: Serialization ---------------- #
    def _serialize_result(self, rr: ReadResultLike) -> bytes:
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
        is_summary = not has_logits  # i.e., ReadResult

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
                "variable_region_range": getattr(rr, "variable_region_range", (-1, -1)),  # None = not predicted
                "smoothed_variable_region_range": getattr(rr, "smoothed_variable_region_range", (-1, -1)),
                "fragmented": _optional_bool(getattr(rr, "fragmented", False)),
                "cropped": bool(getattr(rr, "cropped", False)),
            }
            s = json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            buf.write(self._S_U32.pack(len(s)))
            buf.write(s)
            return buf.getvalue()

        # ---- FULL PAYLOAD (logits blocks) ----
        # cropped flag (format >= 101). Summary records carry this in their JSON.
        buf.write(struct.pack("<B", 1 if bool(getattr(rr, "cropped", False)) else 0))

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
    """Manages writing to ZIR files with optional sharding and comprehensive path handling.

    Each run writes its shards as ``<name>.zir.partial`` inside its own hidden
    directory, ``<output_dir>/.tmp_<timestamp>_<id>/``. Only :meth:`close`, once every
    shard is finished, moves them into the output directory under their final
    ``.zir`` names and removes the temp directory. So the output directory only ever
    holds complete runs, and concurrent runs never write into each other's files.

    :meth:`abort` closes the shards cleanly and leaves the temp directory in place,
    so the reads written so far can be inspected (``ZIRReader`` opens an explicitly
    named ``.zir.partial``; directory scans skip them).

    Existing files with the same final names are overwritten on close; that is the
    caller's responsibility (a warning is logged when they exist).

    With ``staging_dir``, the shards go into that existing temp dir instead, and
    whoever created it finalizes it (see :func:`finalize_tmp_dir`): ``close()`` and
    ``abort()`` then only close the files, and no startup warnings are logged. This
    lets several managers (e.g. one per GPU worker) make up one run.

    Use as a context manager to get close() on success and abort() on error.
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        metadata: 'InferenceMetadata',
        shard_size: Optional[int] = None,
        staging_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize shard manager with extensive path management.
        
        Args:
            base_path: Base path for output file(s)
                - If shard_size is None: Must be a file path ending with .zir
                - If shard_size is set:
                    - Directory: Files saved as dir/shard0000.zir, dir/shard0001.zir
                    - Path without extension: /path/save -> /path/save_shard0000.zir
                    - Path with extension: Extension ignored, user warned
            metadata: Inference metadata
            shard_size: Records per shard (None for single file)
            staging_dir: Existing temp dir owned by the caller (see class docstring);
                None = create and finalize one per manager
            
        Raises:
            ValueError: If path handling fails
        """
        self.metadata = metadata
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_writer: Optional[ZIRWriter] = None
        self.current_count = 0
        self.total_reads = 0
        self.shard_paths: list = []   # final .zir paths, in shard order

        # Process and validate base path
        self._process_base_path(base_path)
        
        # Create necessary directories
        self._ensure_directories()
        self.owns_tmp_dir = staging_dir is None
        if self.owns_tmp_dir:
            self._warn_existing_outputs()
            self.tmp_dir = make_tmp_dir(self.output_dir, metadata.run_id)
        else:
            self.tmp_dir = Path(staging_dir)
            if not self.tmp_dir.is_dir():
                raise FileNotFoundError(f"Staging dir does not exist: {self.tmp_dir}")

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
    
    def add_result(self, read_result: ReadResultLike) -> None:
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
    
    def _warn_existing_outputs(self) -> None:
        if self.shard_size is None:
            existing = [self.base_path] if self.base_path.exists() else []
        else:
            existing = sorted(self.output_dir.glob(
                self.shard_pattern.replace("{:04d}", "[0-9][0-9][0-9][0-9]")))
        warn_existing_outputs(self.output_dir, existing)

    def _partial(self, path: Path) -> Path:
        return self.tmp_dir / (path.name + PARTIAL_SUFFIX)

    def _open_new_shard(self) -> None:
        """Open a new shard file (under its .partial name)."""
        shard_path = self._get_shard_path()
        self.shard_paths.append(shard_path)
        self.current_writer = ZIRWriter(self._partial(shard_path), self.metadata)
        self.current_writer.__enter__()
        self.current_count = 0

    def _close_current_shard(self) -> None:
        """Close current shard."""
        if self.current_writer:
            self.current_writer.__exit__(None, None, None)
            
            self.current_writer = None
            self.current_shard += 1
    
    def close(self) -> None:
        """Close the open shard; if this manager owns the temp dir, finalize it."""
        self._finish_shards()
        if self.owns_tmp_dir and self.tmp_dir.exists():
            finalize_tmp_dir(self.tmp_dir, self.output_dir)

    def abort(self) -> None:
        """Close the open shard and leave all shards in the temp dir as .partial."""
        self._finish_shards()
        if self.owns_tmp_dir and self.tmp_dir.exists():
            logger.warning(
                "Run did not finish; %d incomplete shard(s) with %d read(s) left in %s",
                len(self.shard_paths), self.total_reads, self.tmp_dir,
            )

    def _finish_shards(self) -> None:
        """Close the open shard and give every shard the final metadata.

        ``num_reads_processed`` becomes the reads written by this manager (all its
        shards); the caller may have set other end-of-run fields (e.g. the time)
        on the shared metadata object before calling close()/abort().
        """
        self.metadata.num_reads_processed = self.total_reads
        self._close_current_shard()
        for final in self.shard_paths:
            partial = self._partial(final)
            if partial.exists():
                rewrite_header(partial, self.metadata)

    def __enter__(self) -> "ZIRShardManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# ---------------- Backend-agnostic writer ---------------- #
def write_zir(
    store: "InferenceStore",
    path: Union[str, Path],
    *,
    shard_size: Optional[int] = None,
) -> Path:
    """Serialize any InferenceStore (in-memory or on-disk) to ZIR.

    Works for both InferenceResults and ZIRReader because it only needs the store's
    ``metadata`` and an iteration over its reads. Iterating a ZIRReader streams reads
    from disk, so re-sharding/re-compressing an archive does not materialize it in RAM.

    Args:
        store: Any object satisfying the InferenceStore protocol.
        path: Output path (file for single, directory/prefix for sharded).
        shard_size: Records per shard, or None for a single file.

    Returns:
        The single-file path (single mode) or the output directory (sharded mode).
    """
    with ZIRShardManager(path, store.metadata, shard_size) as manager:
        for read in store:
            manager.add_result(read)
    return manager.base_path if shard_size is None else manager.output_dir