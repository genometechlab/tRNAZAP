import struct
import json
import os
import uuid
from pathlib import Path
from typing import Union, List, Collection, Set, Optional, Iterator, Generator, Iterable, Dict, Any, TYPE_CHECKING
import numpy as np
import zstandard as zstd
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from .archive_format import (MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE, RECORD_MARKER, 
                             INDEX_MAGIC, FOOTER_MAGIC, ENC_UTF8LEN, ENC_UUID16)
from ...utils import search_path

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, InferenceResults, ReadResult

logger = logging.getLogger(__name__)
PathLike = Union[str, Path, os.PathLike]


class ZIRReader:
    """Optimized ZIR archive reader with dynamic logit support."""
    
    def __init__(self, paths: Union[Path, List[Path]], index: bool = False):
        # Initialize struct cache for better performance
        self._struct_cache = {
            'H': struct.Struct('<H'),
            'i': struct.Struct('<i'),
            'B': struct.Struct('<B'),
            'I': struct.Struct('<I'),
            'ii': struct.Struct('<ii'),
            'II': struct.Struct('<II'),
        }
        
        self._paths: List[Path] = sorted(
            self._collect_dataset(paths, recursive=True, pattern='*.zir', threads=4)
        )
        if not self._paths:
            raise ValueError(f"No ZIR file was found")
        self.is_multi = len(self._paths) > 1

        self._files = []
        self.decompressors = []
        self.record_counts = []
        self.record_count = 0
        metadata_dicts = []
        
        self._tocs = []           # NEW: list[dict or None] parallel to files
        self._toc_index = {}     # NEW: read_id -> (file_idx, offset, size)

        # Use buffered I/O for better performance
        for file_idx, p in enumerate(self._paths):
            f = open(p, 'rb', buffering=65536)  # 64KB buffer
            self._files.append(f)
            metadata_dict, record_count = self._read_header(f)
            self.decompressors.append(zstd.ZstdDecompressor())
            self.record_counts.append(record_count)
            self.record_count += record_count
            metadata_dicts.append(metadata_dict)
            
            # Try footer TOC
            toc = self._read_footer(f)     # NEW
            self._tocs.append(toc)
            if toc:
                for rid, info in toc.items():
                    self._toc_index[rid] = (file_idx, info['offset'], info['size'])

        if len(metadata_dicts) > 1:
            self._check_metadata_compatibility(metadata_dicts)
        self.metadata_dict = metadata_dicts[0]

        self._fallback_index = None
        self._current_file_idx = 0
        
        # Only build legacy scan index if requested AND we have no footer TOC
        if index and not self._toc_index:          # NEW
            self.build_fallback_index()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()

    def __len__(self): return self.record_count
    def __iter__(self): return self.reads()

    def reads(self, selection: Optional[Set[str]] = None) -> Generator["ReadResult", None, None]:
        if selection is not None and not isinstance(selection, set):
            selection = set(selection)

        # Fast path when we have any index and a selection

        if selection and (self._toc_index or self._fallback_index):
            from collections import defaultdict
            hits_by_file = defaultdict(list)

            for rid in selection:
                if self._toc_index:
                    hit = self._toc_index.get(rid)
                    if hit:
                        file_idx, offset, _ = hit
                        hits_by_file[file_idx].append(offset)
                else:
                    info = self._fallback_index.get(rid)
                    if info:
                        hits_by_file[info['file_idx']].append(info['offset'])

            for file_idx, offsets in hits_by_file.items():
                f = self._files[file_idx]
                # optional: offsets.sort()
                for offset in offsets:
                    f.seek(offset)
                    yield self._read_next_record(file_idx)
            return

        # Fallback: sequential scan (no selection, or no index available)
        for file_idx, (f, record_count) in enumerate(zip(self._files, self.record_counts)):
            self._current_file_idx = file_idx
            f.seek(HEADER_SIZE)
            for _ in range(record_count):
                result = self._read_next_record(file_idx)
                if selection is None or result.read_id in selection:
                    yield result


    def _read_next_record(self, file_idx: int) -> "ReadResult":
        """Optimized record reading with single read for header."""
        f = self._files[file_idx]
        
        # Read header in one go
        header = f.read(len(RECORD_MARKER) + 8)
        if len(header) < len(RECORD_MARKER) + 8:
            raise EOFError("Unexpected EOF while reading record header")
        
        if header[:len(RECORD_MARKER)] != RECORD_MARKER:
            raise EOFError(f"Invalid record marker at offset {f.tell() - len(header)}")
        
        compressed_size, uncompressed_size = struct.unpack('<II', header[len(RECORD_MARKER):])
        
        # Read compressed data
        compressed_data = f.read(compressed_size)
        if len(compressed_data) < compressed_size:
            raise EOFError("Unexpected EOF while reading compressed data")
        
        # Decompress
        try:
            data = self.decompressors[file_idx].decompress(compressed_data)
        except zstd.ZstdError as e:
            raise ValueError(f"Decompression failed: {e}")
        
        return self._parse_record(data)

    def _parse_record(self, data: bytes) -> "ReadResult":
        """Optimized record parser using memoryview and cached structs."""
        view = memoryview(data)
        offset = 0
        
        # Read ID
        read_id_len = self._struct_cache['H'].unpack_from(view, offset)[0]
        offset += 2
        read_id = data[offset:offset + read_id_len].decode('utf-8')
        offset += read_id_len
        
        # Read metadata
        num_chunks, chunk_size = self._struct_cache['ii'].unpack_from(view, offset)
        offset += 8
        
        # Read logits count
        num_logits = view[offset]
        offset += 1
        
        logits = {}
        
        for _ in range(num_logits):
            # Key name
            key_len = view[offset]
            offset += 1
            key = data[offset:offset + key_len].decode('utf-8')
            offset += key_len
            
            # Array dimensions
            ndim = view[offset]
            offset += 1
            
            # Optimize for common cases
            if ndim == 1:
                dim = self._struct_cache['I'].unpack_from(view, offset)[0]
                shape = (dim,)
                offset += 4
                total_elements = dim
            elif ndim == 2:
                shape = self._struct_cache['II'].unpack_from(view, offset)
                offset += 8
                total_elements = shape[0] * shape[1]
            else:
                # Cache dynamic formats
                format_key = f'{ndim}I'
                if format_key not in self._struct_cache:
                    self._struct_cache[format_key] = struct.Struct(f'<{format_key}')
                shape = self._struct_cache[format_key].unpack_from(view, offset)
                offset += 4 * ndim
                total_elements = np.prod(shape)
            
            # Read array data directly from memoryview
            array = np.frombuffer(view, dtype=np.float32, count=total_elements, offset=offset)
            
            # Only reshape if multi-dimensional
            if ndim > 1:
                array = array.reshape(shape)
            
            logits[key] = array
            offset += total_elements * 4
        
        from ...storages import ReadResult
        return ReadResult(read_id=read_id, _logits=logits, num_chunks=num_chunks, chunk_size=chunk_size)


    def _safe_unpack(self, file, fmt: str, label: str) -> int:
        size = struct.calcsize(fmt)
        data = file.read(size)
        if len(data) < size:
            raise EOFError(f"Unexpected EOF while reading {label}")
        return struct.unpack(fmt, data)[0]

    def _read_header(self, f) -> tuple[dict, int]:
        magic = f.read(len(MAGIC_BYTES))
        if magic != MAGIC_BYTES:
            raise ValueError("Invalid magic bytes")

        version = self._safe_unpack(f, '<I', 'format version')
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format version {version}")

        record_count = self._safe_unpack(f, '<I', 'record count')
        meta_len = self._safe_unpack(f, '<I', 'metadata length')
        metadata = json.loads(f.read(meta_len).decode('utf-8'))

        f.seek(HEADER_SIZE)
        return metadata, record_count
    
    def _read_footer(self, f):
        """Read footer TOC if present. Returns dict[read_id] -> {offset,size} or None."""
        # Seek to end and read trailer [FOOTER_MAGIC][footer_start:u64]
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size < 16:
            return None
        f.seek(file_size - 16)
        trailer = f.read(16)
        if trailer[:8] != FOOTER_MAGIC:
            return None
        (footer_start,) = struct.unpack('<Q', trailer[8:16])
        if not (0 <= footer_start <= file_size - 16):
            return None

        # Parse index header: [INDEX_MAGIC][enc:u16][N:u32]
        f.seek(footer_start)
        if f.read(8) != INDEX_MAGIC:
            return None
        (enc,) = struct.unpack('<H', f.read(2))
        (n,)   = struct.unpack('<I', f.read(4))

        idx = {}
        if enc == ENC_UUID16:
            for _ in range(n):
                id_bytes = f.read(16)
                (offset,) = struct.unpack('<Q', f.read(8))
                (size32,) = struct.unpack('<I', f.read(4))
                rid = str(uuid.UUID(bytes=id_bytes))
                idx[rid] = {'offset': offset, 'size': size32}
        elif enc == ENC_UTF8LEN:
            for _ in range(n):
                (id_len,) = struct.unpack('<H', f.read(2))
                rid = f.read(id_len).decode('utf-8')
                (offset,) = struct.unpack('<Q', f.read(8))
                (size32,) = struct.unpack('<I', f.read(4))
                idx[rid] = {'offset': offset, 'size': size32}
        else:
            return None

        return idx


    def _collect_dataset(
        self, paths: Union[PathLike, Collection[PathLike]],
        recursive: bool, pattern: str, threads: int
    ) -> Set[Path]:
        if isinstance(paths, (str, Path, os.PathLike)):
            paths = [paths]

        paths = [Path(p) for p in paths]
        collected: Set[Path] = set()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            search = partial(search_path, recursive=recursive, patterns=[pattern])
            for found in executor.map(search, paths):
                collected.update(found)

        return collected

    def _check_metadata_compatibility(self, metas: List[Dict[str, Any]]):
        ref = metas[0]
        must_match = ['model_name', 'model_type', 'chunk_size']
        # Removed 'num_classes', 'num_classes_seq2seq' as these might not exist with dynamic logits

        for i, meta in enumerate(metas[1:], 1):
            for field in must_match:
                if ref.get(field) != meta.get(field):
                    raise ValueError(f"Metadata mismatch in {self._paths[i]}: {field} differs")

        # Optional warnings
        for field in ['batch_size', 'device', 'float_dtype']:
            values = [m.get(field) for m in metas]
            if len(set(values)) > 1:
                logger.warning(f"Field '{field}' differs across archives: {values}")

    def build_fallback_index(self) -> None:
        """Build index using parallel processing for large archives."""
        if self._fallback_index is not None:
            return
        
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        self._fallback_index = {}
        index_lock = threading.Lock()
        
        def index_file(file_idx: int, path: Path, record_count: int):
            """Index a single file in a worker thread."""
            local_index = {}
            
            with open(path, 'rb', buffering=65536) as f:
                decompressor = zstd.ZstdDecompressor()
                f.seek(HEADER_SIZE)
                
                for _ in range(record_count):
                    pos = f.tell()
                    
                    # Read header
                    header = f.read(len(RECORD_MARKER) + 8)
                    if len(header) < len(RECORD_MARKER) + 8:
                        break
                    
                    if header[:len(RECORD_MARKER)] != RECORD_MARKER:
                        raise ValueError(f"Invalid record marker at {pos}")
                    
                    compressed_size, uncompressed_size = struct.unpack('<II', header[len(RECORD_MARKER):])
                    
                    # Read compressed data
                    compressed = f.read(compressed_size)
                    
                    # Only decompress first part to get read ID
                    data = decompressor.decompress(compressed)
                    read_id_len = struct.unpack('<H', data[:2])[0]
                    read_id = data[2:2 + read_id_len].decode('utf-8')
                    
                    local_index[read_id] = {
                        'file_idx': file_idx,
                        'offset': pos,
                        'record_size': len(header) + compressed_size,
                    }
            
            # Merge into global index
            with index_lock:
                self._fallback_index.update(local_index)
        
        # Index files in parallel
        with ThreadPoolExecutor(max_workers=min(4, len(self._paths))) as executor:
            futures = []
            for idx, (path, count) in enumerate(zip(self._paths, self.record_counts)):
                future = executor.submit(index_file, idx, path, count)
                futures.append(future)
            
            for future in futures:
                future.result()

    def get_read(self, read_id: str) -> "ReadResult":
        hit = self._toc_index.get(read_id)           # NEW
        if hit:
            file_idx, offset, _size = hit
            f = self._files[file_idx]
            f.seek(offset)
            return self._read_next_record(file_idx)

        # Fallback to legacy index
        if self._fallback_index is None:
            self.build_fallback_index()
        if read_id not in self._fallback_index:
            raise KeyError(f"Read ID '{read_id}' not found")
        info = self._fallback_index[read_id]
        f = self._files[info['file_idx']]
        f.seek(info['offset'])
        return self._read_next_record(info['file_idx'])

    def get_path(self, read_id: str) -> Path:
        hit = self._toc_index.get(read_id)           # NEW
        if hit:
            file_idx, _, _ = hit
            return self._paths[file_idx]
        if self._fallback_index is None:
            self.build_fallback_index()
        return self._paths[self._fallback_index[read_id]['file_idx']]

    def __contains__(self, read_id: str) -> bool:
        if self._toc_index:                          # NEW
            return read_id in self._toc_index
        if self._fallback_index is None:
            self.build_fallback_index()
        return read_id in self._fallback_index

    @property
    def read_ids(self) -> List[str]:
        if self._toc_index:                          # NEW
            return list(self._toc_index.keys())
        if self._fallback_index is None:
            self.build_fallback_index()
        return list(self._fallback_index.keys())


    @property
    def metadata(self) -> "InferenceMetadata":
        from ...storages import InferenceMetadata
        meta_copy = self.metadata_dict.copy()
        return InferenceMetadata(**meta_copy)

    def to_inference_results(self) -> "InferenceResults":
        from ...storages import InferenceResults, InferenceMetadata
        meta_copy = self.metadata_dict.copy()
        if meta_copy.get('pod5_paths'):
            meta_copy['pod5_paths'] = set(meta_copy['pod5_paths'])

        metadata = InferenceMetadata(**meta_copy)
        results = InferenceResults(metadata=metadata)
        for r in self:
            results._add_result(r)
        return results

    def summary(self) -> dict:
        return {
            'num_archives': len(self._paths),
            'paths': [str(p) for p in self._paths],
            'record_counts': self.record_counts,
            'total_records': self.record_count,
            'indexed': self._fallback_index is not None,
            'index_size': len(self._fallback_index) if self._fallback_index else 0,
            'format_version': FORMAT_VERSION,
            'model_name': self.metadata_dict.get('model_name'),
        }