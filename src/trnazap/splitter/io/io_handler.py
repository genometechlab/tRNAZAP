# file: io_handler.py
from abc import ABC, abstractmethod
from typing import Dict, Protocol, Type, TypeVar, Union, Any
from pathlib import Path
import logging
import pickle

from .archive import ZIRWriter, ZIRReader
logger = logging.getLogger(__name__)

T = TypeVar('T')

class Serializable(Protocol):
    """Protocol for objects that can be serialized."""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T: ...

class IOHandler(ABC):
    """Abstract base class for I/O handlers."""
    
    @abstractmethod
    def save(self, obj: Any, path: Path) -> None:
        """Save object to file."""
        pass
    
    @abstractmethod
    def load(self, path: Path, cls: Type[T]) -> T:
        """Load object from file."""
        pass

class ParquetHandler(IOHandler):
    """Handler for Parquet I/O operations."""
    
    def __init__(self, compression: str = 'zstd', compression_level: int = 3):
        self.compression = compression
        self.compression_level = compression_level
    
    def save(self, obj: Any, path: Path) -> None:
        """Save object to Parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        import json
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(obj, '_to_parquet_records'):
            # Custom serialization method
            records, metadata = obj._to_parquet_records()
        else:
            raise TypeError(f"Object of type {type(obj)} cannot be saved to Parquet")
        
        # Create table
        table = pa.Table.from_pylist(records)
        
        # Add metadata if provided
        if metadata:
            schema_metadata = {
                k: json.dumps(v, default=str) if not isinstance(v, str) else v
                for k, v in metadata.items()
            }
            table = table.replace_schema_metadata(schema_metadata)
        
        # Write with compression
        pq.write_table(
            table, 
            path, 
            compression=self.compression,
            compression_level=self.compression_level
        )
    
    def load(self, path: Path, cls: Type[T]) -> T:
        """Load object from Parquet file."""
        import pyarrow.parquet as pq
        import json
        
        table = pq.read_table(path)
        
        # Extract metadata
        metadata = {}
        if table.schema.metadata:
            for key, value in table.schema.metadata.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                try:
                    metadata[key_str] = json.loads(value_str)
                except json.JSONDecodeError:
                    metadata[key_str] = value_str
        
        # Convert to records
        records = table.to_pylist()
        
        # Use class method to reconstruct
        if hasattr(cls, '_from_parquet_records'):
            return cls._from_parquet_records(records, metadata)
        else:
            raise TypeError(f"Class {cls} cannot be loaded from Parquet")

class PickleHandler(IOHandler):
    """Handler for Pickle I/O operations."""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def save(self, obj: Any, path: Path) -> None:
        """Save object to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=self.protocol)
    
    def load(self, path: Path, cls: Type[T]) -> T:
        """Load object from pickle file."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
    

class ZIRHandler(IOHandler):
    """Handler for ZIR archive I/O operations."""
    
    def save(self, obj: Any, path: Path) -> None:
        """Save to ZIR archive format."""
        if not hasattr(obj, '_to_zir'):
            raise TypeError(f"Type {type(obj).__name__} does not support ZIR format")
        obj._to_zir(path)
    
    def load(self, path: Path, cls: Type[T]) -> T:
        """Load from ZIR archive format."""
        if not hasattr(cls, '_from_zir'):
            raise TypeError(f"Type {cls.__name__} does not support ZIR format")
        return cls._from_zir(path)