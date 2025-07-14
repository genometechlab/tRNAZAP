# file: io_manager.py
from typing import Type, TypeVar, Dict, Union, Optional, Tuple, List, Any, Protocol, runtime_checkable
from pathlib import Path
import logging

from .io_config import FileFormat, FormatConfig
from .io_handler import IOHandler, ParquetHandler, PickleHandler, ZIRHandler

logger = logging.getLogger(__name__)

T = TypeVar('T')

@runtime_checkable
class ParquetSerializable(Protocol):
    """Protocol for objects that can be serialized to Parquet."""
    def _to_parquet_records(self) -> Tuple[List[Dict], Dict[str, Any]]: ...
    @classmethod
    def _from_parquet_records(cls: Type[T], records: List[Dict], metadata: Dict[str, Any]) -> T: ...

class IOManager:
    """Centralized I/O manager for all serialization operations."""
    
    _handlers: Dict[FileFormat, IOHandler] = {}
    
    def __init__(self):
        """Initialize with default handlers."""
        self._handlers = {
            FileFormat.PARQUET: ParquetHandler(),
            FileFormat.PICKLE: PickleHandler(),
            FileFormat.ZIR: ZIRHandler()  # Add ZIR handler
        }
    
    def register_handler(self, format: FileFormat, handler: IOHandler) -> None:
        """Register a custom handler for a format."""
        self._handlers[format] = handler
    
    def save(self, obj: Any, path: Union[str, Path], format: Optional[FileFormat] = None) -> Path:
        """
        Save object to file with appropriate format.
        
        Args:
            obj: Object to save
            path: Output path
            format: File format (auto-detected if None)
            
        Returns:
            Path: Actual path where file was saved
        """
        path = Path(path)
        
        # Determine format
        if format is None:
            try:
                format = FormatConfig.detect_format(path)
            except ValueError:
                format = self._infer_format(obj)
        
        # Ensure proper extension
        path = FormatConfig.ensure_extension(path, format)
        
        # Get handler and save
        handler = self._get_handler(format)
        handler.save(obj, path)
        
        logger.info(f"Saved {type(obj).__name__} to {path} (format: {format.value})")
        return path
    
    def load(self, path: Union[str, Path], cls: Type[T]) -> T:
        """
        Load object from file.
        
        Args:
            path: Input file path
            cls: Expected class type
            
        Returns:
            Loaded object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Detect format
        format = FormatConfig.detect_format(path)
        
        # Get handler and load
        handler = self._get_handler(format)
        obj = handler.load(path, cls)
        
        logger.info(f"Loaded {cls.__name__} from {path} (format: {format.value})")
        return obj
    
    def _get_handler(self, format: FileFormat) -> IOHandler:
        """Get handler for format."""
        if format not in self._handlers:
            raise ValueError(f"No handler registered for format: {format}")
        return self._handlers[format]
    
    def _infer_format(self, obj: Any) -> FileFormat:
        """Infer best format for object type."""
        if isinstance(obj, ParquetSerializable):
            return FileFormat.PARQUET
        return FileFormat.PICKLE  # Fallback

# Global instance
global_io_manager = IOManager()