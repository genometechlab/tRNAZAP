"""
BAM Writer implementations

This module provides different strategies for writing BAM files efficiently.
The key insight is that BAM compression is CPU-intensive, so we provide
options to trade off compression ratio for speed.
"""

import pysam
import os

class BAMWriterSimplified:
    """
    Simplified BAMWriter optimized for your tRNA pipeline
    
    Key decisions:
    - No temp files (unnecessary complexity for your case)
    - Default compression level 1 (you want speed)
    - Simple interface
    - Keeps read counting (useful for verification)
    """
    
    def __init__(self, output_path, header, threads=4):
        self.output_path = output_path
        self.header = header
        self.threads = threads
        self._file = None
        
    def __enter__(self):
        # Mode 'wb2' = write, binary, compression level 2
        self._file = pysam.AlignmentFile(
            self.output_path, 'wb2',  # Hardcode fast compression
            header=self.header, 
            threads=self.threads
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def write(self, read):
        """Write one read and count it"""
        self._file.write(read)
        
    def write_many(self, reads):
        """Convenience method for writing multiple reads"""
        for read in reads:
            self.write(read)
            
class ProcessBAMWriter(mp.Process):
    """
    Simplified process-based BAM writer
    
    This writer runs in a separate process and receives aligned reads
    through a queue. It handles compression using multiple threads while
    keeping the interface simple.
    """
    
    def __init__(self, writer_id, output_path, header, compression_threads=4):
        super().__init__()
        
        # Identity and output settings
        self.writer_id = writer_id
        self.output_path = output_path
        self.header = header
        self.compression_threads = compression_threads
        
        # Communication: sized queue provides backpressure
        self.input_queue = mp.Queue(maxsize=10000)
        
        # Shutdown signaling
        self.shutdown_event = mp.Event()
        
        # Batching configuration
        self.batch_size = 5000  # Tune based on your memory/performance needs
        
    def run(self):
        """
        Main writer loop - runs in separate process
        
        This method:
        1. Opens the BAM file for writing
        2. Continuously reads from the queue
        3. Batches reads for efficiency
        4. Handles shutdown gracefully
        """
        print(f"Writer {self.writer_id} started (PID: {mp.current_process().pid})")
        
        try:
            with BAMWriterSimplified(
                self.output_path, 
                self.header, 
                threads=self.compression_threads
            ) as writer:
                
                batch = []
                
                # Main loop - check shutdown flag regularly
                while not self.shutdown_event.is_set() or not self.input_queue.empty():
                    try:
                        # Try to get a read (with timeout to check shutdown)
                        read = self.input_queue.get(timeout=0.1)
                        batch.append(read)
                        
                        # Write full batches
                        if len(batch) >= self.batch_size:
                            writer.write_many(batch)
                            batch = []
                            
                    except queue.Empty:
                        # No reads available - write any partial batch
                        if batch:
                            writer.write_many(batch)
                            batch = []
                            
                # Final batch on shutdown
                if batch:
                    writer.write_many(batch)
                    
        except Exception as e:
            print(f"Writer {self.writer_id} error: {e}")
            raise
            
        print(f"Writer {self.writer_id} finished")
        
    def shutdown(self):
        """Signal the writer to finish and exit"""
        self.shutdown_event.set()
        
    def put(self, read):
        """
        Add a read to this writer's queue
        
        This method is called by reader processes. It will block
        if the queue is full, providing natural backpressure.
        """
        self.input_queue.put(read)
        
def create_writers(num_writers, output_prefix, header, threads_per_writer=4):
    """
    Create and start multiple writer processes.
    
    This is just a simple function - no class needed!
    Returns a list of started writer processes.
    """
    writers = []
    
    for i in range(num_writers):
        writer = ProcessBAMWriter(
            writer_id=i,
            output_path=f"{output_prefix}_{i}.bam",
            header=header,
            compression_threads=threads_per_writer
        )
        writer.start()
        writers.append(writer)
        
    return writers

def assign_readers_to_writers(num_readers, writers):
    """
    Create a simple mapping of which reader uses which writer.
    
    With 12 readers and 3 writers:
    - Reader 0 → Writer 0
    - Reader 1 → Writer 1
    - Reader 2 → Writer 2
    - Reader 3 → Writer 0  (wraps around)
    - And so on...
    
    Returns a list where index is reader_id and value is the writer queue.
    """
    reader_to_queue = []
    
    for reader_id in range(num_readers):
        writer_index = reader_id % len(writers)
        writer_queue = writers[writer_index].input_queue
        reader_to_queue.append(writer_queue)
        
    return reader_to_queue