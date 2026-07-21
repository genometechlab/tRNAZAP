"""Binary format specification for lazy inference archives."""

# File format constants
MAGIC_BYTES = b'ZIR\x00'
# Version is stored as a uint32, encoded as major*100 + minor.
# v1 (legacy) wrote the literal value 1. v101 == "1.01" adds the per-record
# `cropped` flag; v1 files remain readable (cropped defaults to False).
FORMAT_VERSION = 101
HEADER_SIZE = 65536

# Compression
COMPRESSION_ALGO = 'zstd'  # Fast compression with good ratio
COMPRESSION_LEVEL = 3      # Balance between speed and size

# Record structure markers
RECORD_MARKER = b'REC\x00'  # Marks start of each record
ARRAY_TYPE_CLASSIFICATION = 1
ARRAY_TYPE_SEQ2SEQ = 2
BUFFER_SIZE = 1 << 20 # 1 MiB
PREVIEW_MAX = 512

# Record structure documentation:
"""
Record structure (in order):
1. Record marker (4 bytes) - to detect corruption/alignment
2. Compressed size (4 bytes uint32)
3. Uncompressed size (4 bytes uint32) 
4. Compressed data containing:
   - Read ID length (2 bytes uint16)
   - Read ID (variable string)
   - Num chunks (4 bytes int32)
   - Chunk size (4 bytes int32)
   - Record kind (1 byte uint8): 0 = full/logits, 1 = summary JSON
   - Cropped flag (1 byte uint8, full records only, format >= 101)
   - Classification array info + data
   - Seq2seq array info + data
"""