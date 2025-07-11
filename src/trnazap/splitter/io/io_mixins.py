from typing import Union, Optional, Type, TypeVar, List
from pathlib import Path

from .io_config import FileFormat
from .io_manager import global_io_manager

T = TypeVar('T')

# file: io_mixins.py
class SaveLoadMixin:
    """Mixin providing save/load functionality using IOManager."""
    
    def save(self, path: Union[str, Path], format: Optional[FileFormat] = None) -> Path:
        """Save object to file."""
        return global_io_manager.save(self, path, format)
    
    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """Load object from file."""
        return global_io_manager.load(path, cls)

class MultiLoadMixin(SaveLoadMixin):
    """Mixin for classes that support loading and merging multiple files."""
    
    @classmethod
    def load(cls: Type[T], paths: Union[str, Path, List[Union[str, Path]]]) -> T:
        """Load and optionally merge multiple files."""
        if isinstance(paths, (str, Path)):
            return global_io_manager.load(paths, cls)
        
        # Multiple files - load and merge
        if not paths:
            raise ValueError("No paths provided")
        
        results = [global_io_manager.load(path, cls) for path in paths]
        
        if len(results) == 1:
            return results[0]
        
        # Assume class has merge method
        if hasattr(cls, 'merge'):
            return cls.merge(*results)
        else:
            raise TypeError(f"{cls.__name__} does not support merging multiple files")