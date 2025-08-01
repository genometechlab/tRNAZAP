import struct
import json
import os
from pathlib import Path
from typing import Union, List, Collection, Set, Optional, Iterator, Generator, Iterable, Dict, Any, TYPE_CHECKING
import numpy as np
import zstandard as zstd
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from .archive_format import MAGIC_BYTES, FORMAT_VERSION, HEADER_SIZE, RECORD_MARKER
from ...utils import search_path

if TYPE_CHECKING:
    from ...storages import InferenceMetadata, InferenceResults, ReadResult

logger = logging.getLogger(__name__)
PathLike = Union[str, Path, os.PathLike]


class ZIRReader:
    """Read inference results from one or more ZIR archive files with dynamic logit support."""

    def __init__(self, paths: Union[Path, List[Path]], index: bool = False):
        self._paths: List[Path] = sorted(
            self._collect_dataset(paths, recursive=True, pattern='*.zir', threads=4)
        )
        self.is_multi = len(self._paths) > 1

        self._files = []
        self.decompressors = []
        self.record_counts = []
        self.record_count = 0
        metadata_dicts = []

        for p in self._paths:
            f = open(p, 'rb')
            self._files.append(f)
            metadata_dict, record_count = self._read_header(f)
            self.decompressors.append(zstd.ZstdDecompressor())
            self.record_counts.append(record_count)
            self.record_count += record_count
            metadata_dicts.append(metadata_dict)

        if len(metadata_dicts) > 1:
            self._check_metadata_compatibility(metadata_dicts)
        self.metadata_dict = metadata_dicts[0]

        self._index = None
        self._current_file_idx = 0

        if index:
            self.build_index()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()

    def __len__(self): return self.record_count
    def __iter__(self): return self.reads()

    def reads(self, selection: Optional[Iterable[str]] = None) -> Generator["ReadResult", None, None]:
        for file_idx, (f, record_count) in enumerate(zip(self._files, self.record_counts)):
            self._current_file_idx = file_idx
            f.seek(HEADER_SIZE)

            for _ in range(record_count):
                result = self._read_next_record(file_idx)
                if selection and result.read_id not in selection:
                    continue
                yield result

    def _read_next_record(self, file_idx: int) -> "ReadResult":
        f = self._files[file_idx]

        marker = f.read(len(RECORD_MARKER))
        if marker != RECORD_MARKER:
            raise EOFError(f"Missing or invalid record marker at offset {f.tell()}")

        compressed_size = self._safe_unpack(f, '<I', "compressed size")
        uncompressed_size = self._safe_unpack(f, '<I', "uncompressed size")

        compressed_data = f.read(compressed_size)
        if len(compressed_data) < compressed_size:
            raise EOFError(f"Unexpected EOF while reading compressed data at offset {f.tell()}")

        try:
            data = self.decompressors[file_idx].decompress(compressed_data)
        except zstd.ZstdError as e:
            raise ValueError(f"Decompression failed: {e}")

        if len(data) != uncompressed_size:
            raise ValueError(f"Decompressed size mismatch: expected {uncompressed_size}, got {len(data)}")

        return self._parse_record(data)

    def _parse_record(self, data: bytes) -> "ReadResult":
        """Parse record with dynamic logit support."""
        offset = 0

        # Read ID
        read_id_len = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        read_id = data[offset:offset + read_id_len].decode('utf-8')
        offset += read_id_len

        # Basic metadata
        num_chunks = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        chunk_size = struct.unpack_from('<i', data, offset)[0]
        offset += 4

        logits = {}

        # Check if this is the new format (has logit count) or old format
        # We can detect by checking if the next byte is reasonable
        next_byte = struct.unpack_from('<B', data, offset)[0]
        
        if next_byte <= 20:  # Reasonable number of logit keys (new format)
            # NEW FORMAT: Read dynamic logits
            num_logits = next_byte
            offset += 1
            
            for _ in range(num_logits):
                # Read key name
                key_len = struct.unpack_from('<B', data, offset)[0]
                offset += 1
                key = data[offset:offset + key_len].decode('utf-8')
                offset += key_len
                
                # Read array info
                ndim = struct.unpack_from('<B', data, offset)[0]
                offset += 1
                
                # Read shape
                shape = []
                for _ in range(ndim):
                    dim = struct.unpack_from('<I', data, offset)[0]
                    shape.append(dim)
                    offset += 4
                
                # Read data
                total_elements = np.prod(shape) if shape else 1
                array = np.frombuffer(data, dtype='float32', count=total_elements, offset=offset)
                if shape:
                    array = array.reshape(shape)
                
                logits[key] = array
                offset += total_elements * 4
                
        else:
            # OLD FORMAT: Handle legacy classification/segmentation format
            # Reset and read using old logic
            offset -= 1  # Go back one byte
            
            has_cls = struct.unpack_from('<B', data, offset)[0]
            offset += 1
            if has_cls:
                length = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                logits['seq_class'] = np.frombuffer(data, dtype='float32', count=length, offset=offset)
                offset += length * 4

            has_seq = struct.unpack_from('<B', data, offset)[0]
            offset += 1
            if has_seq:
                T, D = struct.unpack_from('<II', data, offset)
                offset += 8
                logits['seq2seq'] = np.frombuffer(data, dtype='float32', count=T * D, offset=offset).reshape((T, D))

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

    def build_index(self) -> None:
        if self._index is not None:
            return

        self._index = {}
        for file_idx, (f, count) in enumerate(zip(self._files, self.record_counts)):
            f.seek(HEADER_SIZE)
            for _ in range(count):
                pos = f.tell()
                marker = f.read(len(RECORD_MARKER))
                if marker != RECORD_MARKER:
                    raise ValueError(f"Invalid record marker at {pos}")

                compressed_size = self._safe_unpack(f, '<I', 'compressed size')
                uncompressed_size = self._safe_unpack(f, '<I', 'uncompressed size')
                compressed = f.read(compressed_size)

                data = self.decompressors[file_idx].decompress(compressed)
                read_id_len = struct.unpack_from('<H', data, 0)[0]
                read_id = data[2:2 + read_id_len].decode('utf-8')

                self._index[read_id] = {
                    'file_idx': file_idx,
                    'offset': pos,
                    'record_size': len(RECORD_MARKER) + 4 + 4 + compressed_size,
                }

    def get_read(self, read_id: str) -> "ReadResult":
        if self._index is None:
            self.build_index()

        if read_id not in self._index:
            raise KeyError(f"Read ID '{read_id}' not found")

        info = self._index[read_id]
        f = self._files[info['file_idx']]
        f.seek(info['offset'])
        return self._read_next_record(info['file_idx'])

    def get_path(self, read_id: str) -> Path:
        if self._index is None:
            self.build_index()

        return self._paths[self._index[read_id]['file_idx']]

    def __contains__(self, read_id: str) -> bool:
        if self._index is None:
            self.build_index()
        return read_id in self._index

    @property
    def read_ids(self) -> List[str]:
        if self._index is None:
            self.build_index()
        return list(self._index.keys())

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
            'indexed': self._index is not None,
            'index_size': len(self._index) if self._index else 0,
            'format_version': FORMAT_VERSION,
            'model_name': self.metadata_dict.get('model_name'),
        }