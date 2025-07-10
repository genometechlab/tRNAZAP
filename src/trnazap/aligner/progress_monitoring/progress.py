import struct
import threading
import time
from multiprocessing import Pool, shared_memory

from tqdm import tqdm

"""
Progress Monitoring Bars for tRNA_zap Alignment Module.

This module utilizes a shared counter between each of the subprocesses
to track how much progress has been made.
"""


def create_shared_counter():
    """Create a shared memory block to hold a single integer counter."""
    # Create 8 bytes of shared memory (enough for a 64-bit integer)
    shm = shared_memory.SharedMemory(create=True, size=8)
    # Initialize counter to 0
    struct.pack_into("Q", shm.buf, 0, 0)
    return shm


def get_counter_value(shm_name):
    """Read the current counter value from shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    value = struct.unpack_from("Q", shm.buf, 0)[0]
    shm.close()
    return value


def increment_counter(shm_name, amount=1):
    """Add to the counter in shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    current = struct.unpack_from("Q", shm.buf, 0)[0]
    struct.pack_into("Q", shm.buf, 0, current + amount)
    shm.close()


def monitor_progress(shm_name, total_work):
    """Watch the shared counter and update tqdm."""
    pbar = tqdm(total=total_work, desc="Progress")

    last_value = 0
    while last_value < total_work:
        current_value = get_counter_value(shm_name)

        # Update progress bar if there's been progress
        if current_value > last_value:
            pbar.update(current_value - last_value)
            last_value = current_value

        # Check every 100ms
        time.sleep(0.005)

    pbar.close()


def create_monitor(counter_name, total_work):
    monitor_thread = threading.Thread(
        target=monitor_progress, args=(counter_name, total_work)
    )
    return monitor_thread
