"""Binary format specification for lazy inference archives."""

# File format constants
MAGIC_BYTES = b'ZIR\x00'
FORMAT_VERSION = 1
HEADER_SIZE = 65536 

# Compression
COMPRESSION_ALGO = 'zstd'  # Fast compression with good ratio
COMPRESSION_LEVEL = 3      # Balance between speed and size

# Record structure markers
RECORD_MARKER = b'REC\x00'  # Marks start of each record
ARRAY_TYPE_CLASSIFICATION = 1
ARRAY_TYPE_SEQ2SEQ = 2

# TOC
INDEX_MAGIC  = b"ZIRINDEX"
FOOTER_MAGIC = b"ZIRFOOT1"
ENC_UUID16   = 1
ENC_UTF8LEN  = 2

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
   - Classification array info + data
   - Seq2seq array info + data
"""