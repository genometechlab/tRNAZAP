# file: io_config.py
from enum import Enum
from typing import Set, Dict
from pathlib import Path

class FileFormat(Enum):
    """Supported file formats for serialization."""
    PARQUET = "parquet"
    PICKLE = "pickle"
    ZIR = "zir"  # Archive file format

class FormatConfig:
    """Configuration for file formats."""
    
    # Supported extensions for each format
    EXTENSIONS: Dict[FileFormat, Set[str]] = {
        FileFormat.PARQUET: {'.parquet', '.pq', '.parq'},
        FileFormat.PICKLE: {'.pkl', '.pickle'},
        FileFormat.ZIR: {'.zir'}
    }
    
    # Default extension for each format
    DEFAULT_EXTENSION: Dict[FileFormat, str] = {
        FileFormat.PARQUET: '.parquet',
        FileFormat.PICKLE: '.pkl',
        FileFormat.ZIR: '.zir'
    }
    
    # Magic bytes for format detection
    MAGIC_BYTES: Dict[bytes, FileFormat] = {
        b'PAR1': FileFormat.PARQUET,
        b'ZIR\x00': FileFormat.ZIR  # Add ZIR magic bytes
    }
    
    @classmethod
    def detect_format(cls, path: Path) -> FileFormat:
        """
        Detect file format by extension or magic bytes.
        
        Args:
            path: File path
            
        Returns:
            Detected FileFormat
            
        Raises:
            ValueError: If format cannot be determined
        """
        # First try by extension
        suffix = path.suffix.lower()
        for format_type, extensions in cls.EXTENSIONS.items():
            if suffix in extensions:
                return format_type
        
        # Try by magic bytes if file exists
        if path.exists():
            with open(path, 'rb') as f:
                magic = f.read(4)
                for magic_bytes, format_type in cls.MAGIC_BYTES.items():
                    if magic.startswith(magic_bytes):
                        return format_type
        
        raise ValueError(f"Cannot determine format for file: {path}")
    
    @classmethod
    def ensure_extension(cls, path: Path, format_type: FileFormat) -> Path:
        """
        Ensure path has a valid extension for the format.
        
        Args:
            path: Original path
            format_type: Desired format
            
        Returns:
            Path with appropriate extension
        """
        if path.suffix.lower() in cls.EXTENSIONS[format_type]:
            return path
        return path.with_suffix(cls.DEFAULT_EXTENSION[format_type])