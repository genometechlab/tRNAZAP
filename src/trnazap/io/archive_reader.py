import struct
import json
import os
import io
from pathlib import Path
from typing import Union, List, Collection, Set, Tuple, Optional, Iterator, Generator, Iterable, Dict, Any, TYPE_CHECKING
import numpy as np
import zstandard as zstd
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from .archive_format import MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE, RECORD_MARKER, PREVIEW_MAX
from ..utils import search_path

if TYPE_CHECKING:
    from ..storages import InferenceMetadata, InferenceResults, ReadResult, ReadResultCompressed
    
ReadResultUnion = Union["ReadResult", "ReadResultCompressed"]

logger = logging.getLogger(__name__)
PathLike = Union[str, Path, os.PathLike]


def _default_search_path(base: Path, *, recursive: bool, patterns: List[str]) -> List[Path]:
    """Simple glob-based fallback search if your project's search_path isn't available."""
    found: List[Path] = []
    for pat in patterns:
        if recursive:
            found.extend(base.rglob(pat))
        else:
            found.extend(base.glob(pat))
    return found


class ZIRReader:
    """High-performance ZIR archive reader with dynamic logits and selection-aware fast paths.

    Design goals:
    - Buffered I/O on open files
    - One decompressor per file (thread-safe to call within a single thread)
    - Partial-decompress index & selection fast paths using zstd's max_output_size
    - Memoryview + cached struct formats for tight parsing
    - Defensive bounds checks against malformed inputs

    Thread-safety: a single ZIRReader instance is *not* thread-safe. If you need parallelism,
    shard across files or create multiple readers.
    """

    __slots__ = (
        "_paths",
        "_files",
        "decompressors",
        "record_counts",
        "record_count",
        "metadata_dict",
        "_index",
        "_S_U16",
        "_S_U32",
        "_S_I32",
        "_S_II",
    )

    def __init__(self, paths: Union[PathLike, Iterable[PathLike]], *, index: bool = False, threads: int = 4) -> None:
        # Resolve path list
        p_list: List[Path] = []
        if isinstance(paths, (str, os.PathLike, Path)):
            paths = [paths]
        for p in paths:
            p = Path(p)
            if p.is_dir():
                try:
                    from ..utils import search_path  # type: ignore
                    found = search_path(p, recursive=True, patterns=["*.zir"])  # returns Iterable[Path]
                except Exception:
                    found = _default_search_path(p, recursive=True, patterns=["*.zir"])  # fallback
                p_list.extend(Path(x) for x in found)
            elif p.is_file() and p.suffix == ".zir":
                p_list.append(p)
        self._paths = sorted(set(p_list))
        if not self._paths:
            raise ValueError("No ZIR file was found")

        # Cached struct packers
        self._S_U16 = struct.Struct("<H")
        self._S_U32 = struct.Struct("<I")
        self._S_I32 = struct.Struct("<i")
        self._S_II = struct.Struct("<II")

        self._files: List[io.BufferedReader] = []
        self.decompressors: List[zstd.ZstdDecompressor] = []
        self.record_counts: List[int] = []
        self.record_count: int = 0
        metadata_dicts: List[Dict[str, Any]] = []

        # Open each archive with buffering and read fixed header
        for p in self._paths:
            raw = open(p, "rb", buffering=0)
            f = io.BufferedReader(raw, buffer_size=1 << 20)  # 1 MiB read buffer
            md, rc = self._read_header(f)
            self._files.append(f)
            self.decompressors.append(zstd.ZstdDecompressor())
            self.record_counts.append(rc)
            self.record_count += rc
            metadata_dicts.append(md)

        # Basic metadata compatibility (allow dynamic logits; require core fields)
        if len(metadata_dicts) > 1:
            self._check_metadata_compatibility(metadata_dicts)
        self.metadata_dict = metadata_dicts[0]

        self._index: Optional[Dict[str, Dict[str, Any]]] = None
        if index:
            self.build_index(threads=threads)

    # ---------------- Context Manager & Basic dunder ---------------- #
    def __enter__(self) -> "ZIRReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    def __len__(self) -> int:
        return self.record_count

    def __iter__(self) -> Generator[ReadResultUnion, None, None]:
        return self.reads()

    # ---------------- Public API ---------------- #
    def reads(self, selection: Optional[Set[str]] = None) -> Generator[ReadResultUnion, None, None]:
        """Iterate reads; if *selection* provided, use fast path to avoid full decompress.

        Args:
            selection: Optional set of read IDs to include.
        """
        if selection is not None and not isinstance(selection, set):
            selection = set(selection)

        for file_idx, (f, rc) in enumerate(zip(self._files, self.record_counts)):
            f.seek(HEADER_SIZE)
            for _ in range(rc):
                pos = f.tell()
                header = f.read(len(RECORD_MARKER) + 8)
                if len(header) < len(RECORD_MARKER) + 8:
                    raise EOFError("Unexpected EOF while reading record header")
                if header[: len(RECORD_MARKER)] != RECORD_MARKER:
                    got = header[: len(RECORD_MARKER)].hex()
                    raise EOFError(f"Invalid record marker at {pos} (got 0x{got})")

                csize, usize = struct.unpack_from('<II', header, len(RECORD_MARKER))
                compressed = f.read(csize)
                if len(compressed) < csize:
                    raise EOFError("Unexpected EOF while reading compressed data")

                if selection is None:
                    # Full parse path
                    data = self._decompress_full(file_idx, compressed, usize)
                    yield self._parse_record(data)
                else:
                    # Peek read_id with partial decompress
                    preview = self.decompressors[file_idx].decompress(compressed, max_output_size=PREVIEW_MAX)
                    if len(preview) < 2:
                        raise ValueError("Corrupt record: cannot read read_id length from preview")
                    rid_len = self._S_U16.unpack_from(preview, 0)[0]
                    end = 2 + rid_len
                    if len(preview) < end:
                        # If preview too small for a very long id, fall back to full decompress
                        data = self._decompress_full(file_idx, compressed, usize)
                        rid_len = self._S_U16.unpack_from(data, 0)[0]
                        rid = data[2 : 2 + rid_len].decode("utf-8")
                    else:
                        rid = preview[2:end].decode("utf-8")

                    if rid in selection:
                        # Now fully decompress and parse
                        data = self._decompress_full(file_idx, compressed, usize)
                        yield self._parse_record(data)
                    # else skip: we already consumed the compressed bytes; continue

    def build_index(self, *, threads: int = 4) -> None:
        if self._index is not None:
            return
        index: Dict[str, Dict[str, Any]] = {}

        def index_one(file_idx: int, path: Path, expected: int) -> Dict[str, Dict[str, Any]]:
            local: Dict[str, Dict[str, Any]] = {}
            with open(path, "rb", buffering=0) as raw:
                f = io.BufferedReader(raw, buffer_size=1 << 20)
                dcmp = zstd.ZstdDecompressor()
                f.seek(HEADER_SIZE)
                for _ in range(expected):
                    pos = f.tell()
                    head = f.read(len(RECORD_MARKER) + 8)
                    if len(head) < len(RECORD_MARKER) + 8:
                        break
                    if head[: len(RECORD_MARKER)] != RECORD_MARKER:
                        got = head[: len(RECORD_MARKER)].hex()
                        raise ValueError(f"Invalid record marker at {pos} in {path} (got 0x{got})")
                    csize, usize = struct.unpack_from("<II", head, len(RECORD_MARKER))
                    comp = f.read(csize)
                    if len(comp) < csize:
                        break
                    prev = dcmp.decompress(comp, max_output_size=PREVIEW_MAX)
                    if len(prev) < 2:
                        continue
                    rid_len = self._S_U16.unpack_from(prev, 0)[0]
                    end = 2 + rid_len
                    if len(prev) < end:
                        # fallback to full
                        data = dcmp.decompress(comp, max_output_size=usize or 0)
                        rid_len = self._S_U16.unpack_from(data, 0)[0]
                        rid = data[2 : 2 + rid_len].decode("utf-8")
                    else:
                        rid = prev[2:end].decode("utf-8")
                    local[rid] = {"file_idx": file_idx, "offset": pos, "record_size": len(head) + csize}
            # merge
            index.update(local)
            return local

        with ThreadPoolExecutor(max_workers=min(threads, max(1, len(self._paths)))) as ex:
            futs = [ex.submit(index_one, i, p, rc) for i, (p, rc) in enumerate(zip(self._paths, self.record_counts))]
            for fu in futs:
                fu.result()
                index.update(fu.result())

        self._index = index

    def get_read(self, read_id: str) -> ReadResultUnion:
        if self._index is None:
            self.build_index()
        assert self._index is not None
        info = self._index.get(read_id)
        if info is None:
            raise KeyError(f"Read ID '{read_id}' not found")
        f = self._files[info["file_idx"]]
        f.seek(info["offset"])
        # read one record at offset
        header = f.read(len(RECORD_MARKER) + 8)
        if len(header) < len(RECORD_MARKER) + 8:
            raise EOFError("Unexpected EOF while reading record header")
        csize = self._S_U32.unpack_from(header, len(RECORD_MARKER))[0]
        comp = f.read(csize)
        data = self.decompressors[info["file_idx"]].decompress(comp, max_output_size=self._S_U32.unpack_from(header, len(RECORD_MARKER)+4)[0])
        return self._parse_record(data)

    def get_path(self, read_id: str) -> Path:
        if self._index is None:
            self.build_index()
        assert self._index is not None
        return self._paths[self._index[read_id]["file_idx"]]

    def __contains__(self, read_id: str) -> bool:
        if self._index is None:
            self.build_index()
        assert self._index is not None
        return read_id in self._index

    @property
    def read_ids(self) -> List[str]:
        if self._index is None:
            self.build_index()
        assert self._index is not None
        return list(self._index.keys())

    @property
    def metadata(self) -> "InferenceMetadata":
        from ..storages import InferenceMetadata  # type: ignore
        return InferenceMetadata(**self.metadata_dict.copy())

    def to_inference_results(self) -> "InferenceResults":
        from ..storages import InferenceResults, InferenceMetadata, ReadResult, ReadResultCompressed  # type: ignore
        meta = self.metadata_dict.copy()
        if meta.get("pod5_paths"):
            meta["pod5_paths"] = set(meta["pod5_paths"])  # restore set if writer serialized it
        md = InferenceMetadata(**meta)
        res = InferenceResults(metadata=md)
        for r in self.reads():
            res.add_result(r)
        return res

    def summary(self) -> Dict[str, Any]:
        return {
            "num_archives": len(self._paths),
            "paths": [str(p) for p in self._paths],
            "record_counts": self.record_counts,
            "total_records": self.record_count,
            "indexed": self._index is not None,
            "index_size": len(self._index) if self._index else 0,
            "format_version": FORMAT_VERSION,
            "model_name": self.metadata_dict.get("model_name"),
        }

    # ---------------- Internal helpers ---------------- #
    def _decompress_full(self, file_idx: int, compressed: bytes, expected_uncompressed: int) -> bytes:
        data = self.decompressors[file_idx].decompress(compressed, max_output_size=expected_uncompressed)
        # Optional sanity: if expected size written by writer, verify
        if expected_uncompressed and len(data) != expected_uncompressed:
            logger.warning(
                "Uncompressed size mismatch: expected %d, got %d", expected_uncompressed, len(data)
            )
        return data

    def _parse_record(self, data: bytes) -> ReadResultUnion:
        view = memoryview(data)
        n = len(view)

        def need(sz: int, at: int) -> None:
            if at + sz > n:
                raise ValueError(f"Corrupt record: need {sz} bytes at {at}, payload size {n}")

        off = 0
        # read_id
        need(2, off)
        rid_len = self._S_U16.unpack_from(view, off)[0]
        off += 2
        need(rid_len, off)
        rid = data[off : off + rid_len].decode("utf-8")
        off += rid_len

        # num_chunks, chunk_size
        need(8, off)
        num_chunks, chunk_size = struct.unpack_from("<ii", view, off)
        off += 8

        # NEW: record_kind (0 = full/logits, 1 = summary JSON)
        need(1, off)
        record_kind = view[off]
        off += 1

        if record_kind == 0:
            # -------- FULL record (logits) --------
            need(1, off)
            num_logits = view[off]
            off += 1
            if num_logits > 255:
                raise ValueError(f"num_logits out of range: {num_logits}")

            logits: Dict[str, np.ndarray] = {}
            for _ in range(num_logits):
                # key
                need(1, off); klen = view[off]; off += 1
                need(klen, off); key = data[off:off+klen].decode("utf-8"); off += klen

                # ndim
                need(1, off); ndim = view[off]; off += 1
                if ndim == 0 or ndim > 255:
                    raise ValueError(f"Unsupported ndim={ndim}")

                # shape
                need(4 * ndim, off)
                if ndim == 1:
                    (d0,) = struct.unpack_from("<I", view, off); off += 4
                    shape = (d0,); total = d0
                elif ndim == 2:
                    d0, d1 = self._S_II.unpack_from(view, off); off += 8
                    shape = (d0, d1); total = d0 * d1
                else:
                    fmt = "<" + "I" * ndim
                    shape = struct.unpack_from(fmt, view, off); off += 4 * ndim
                    total = int(np.prod(shape))
                    if total < 0 or total > 1_000_000_000:
                        raise ValueError(f"Unreasonable array size ({total} elements)")

                # data
                bytes_needed = total * 4
                need(bytes_needed, off)
                arr = np.frombuffer(view, dtype="<f4", count=total, offset=off)
                if ndim > 1:
                    arr = arr.reshape(shape)
                logits[key] = arr
                off += bytes_needed

            from ..storages import ReadResult  # type: ignore
            return ReadResult(read_id=rid, _logits=logits, num_chunks=num_chunks, chunk_size=chunk_size)

        elif record_kind == 1:
            # -------- SUMMARY record (compressed) --------
            need(4, off)
            summary_len = self._S_U32.unpack_from(view, off)[0]
            off += 4
            need(summary_len, off)
            summary = json.loads(data[off:off+summary_len].decode("utf-8"))
            off += summary_len

            from ..storages import ReadResultCompressed  # type: ignore
            top3 = summary.get("top3_classes", [])
            top3_np = np.asarray(top3, dtype=np.int64) if not isinstance(top3, np.ndarray) else top3

            return ReadResultCompressed(
                read_id=rid,
                top3_classes=top3_np,
                variable_region_range=tuple(summary.get("variable_region_range", (-1, -1))),
                fragmented=bool(summary.get("fragmented", False)),
                num_chunks=num_chunks,
                chunk_size=chunk_size,
            )

        else:
            raise ValueError(f"Unknown record_kind={record_kind}")


    def _read_header(self, f: io.BufferedReader) -> Tuple[Dict[str, Any], int]:
        magic = f.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError("Invalid magic bytes")
        ver_b = f.read(4)
        if len(ver_b) < 4:
            raise EOFError("Unexpected EOF while reading format version")
        version = struct.unpack("<I", ver_b)[0]
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format version {version}")

        rc_b = f.read(4)
        if len(rc_b) < 4:
            raise EOFError("Unexpected EOF while reading record count")
        record_count = struct.unpack("<I", rc_b)[0]

        ml_b = f.read(4)
        if len(ml_b) < 4:
            raise EOFError("Unexpected EOF while reading metadata length")
        meta_len = struct.unpack("<I", ml_b)[0]
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        meta = self._validate_meta(meta)

        f.seek(HEADER_SIZE)
        return meta, record_count
    
    def _validate_meta(self, data: dict) -> dict:
        """Migrate metadata dict from old schema to new schema."""
        key_mapping = {
            "num_classes": "num_classification_classes",
            "num_classes_seq2seq": "num_segmentation_classes",
        }
        for old_key, new_key in key_mapping.items():
            if old_key in data and new_key not in data:
                data[new_key] = data.pop(old_key)
        return data

    def _check_metadata_compatibility(self, metas: List[Dict[str, Any]]) -> None:
        ref = metas[0]
        must_match = ["model_name", "model_type", "chunk_size"]
        for i, m in enumerate(metas[1:], 1):
            for field in must_match:
                if ref.get(field) != m.get(field):
                    raise ValueError(f"Metadata mismatch in {self._paths[i]}: {field} differs")
        # Optional: warn on soft fields
        for field in ["float_dtype"]:
            vals = {m.get(field) for m in metas}
            if len(vals) > 1:
                logger.warning("Field '%s' differs across archives: %s", field, list(vals))