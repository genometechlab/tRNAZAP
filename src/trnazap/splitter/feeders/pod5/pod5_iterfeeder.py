import torch.utils.data as data
import numpy as np
import pod5
import torch
from uuid import UUID
from ..signal_utils import load_signal


class Pod5IterDataset(data.Dataset):
    """
    A PyTorch dataset class for handling tRNA nanopore sequencing data stored in POD5 format.
    This dataset loads signals, applies transformations, and prepares labeled training samples.
    """

    def __init__(
        self,
        read_ids: list,
        pod5_paths: str,
        window_size: int,
        step_size: int,
        max_seq_len: int = None,
        batch_size: int = 512,
        transform: callable = None,
        dtype: str = "float32",
        debug: bool = False,
    ):
        """
        Initializes the dataset.

        Args:
            read_ids (list of str): List of read IDs.
            pod5_paths (str): Path to POD5 files.
            window_size (int): Size of the sliding window for chunking the dataset.
            step_size (int): Step size for chunking the dataset.
            batch_size (int): Number of reads per batch (required, default: 512).
            load_labels (bool): Whether to load labels and token run-length encodings (RLE). Defaults to True.
            token_rles (list, optional): Encoded token run-lengths, required if load_labels is True.
            labels (list, optional): Corresponding labels for reads, required if load_labels is True.
            rle_decoder (callable, optional): Function to decode RLE tokens, required if load_labels is True.
            rle_decoder_args (dict, optional): Arguments for the RLE decoder, required if load_labels is True.
            transform (callable, optional): Transformation function for signals.
            dtype (str, optional): Data type of the signal ('single' or 'double'). Defaults to 'single'.
            debug (bool, optional): If True, enables debug mode with error logging. Defaults to False.
        """

        # Set signal data type
        self._get_signal_dtype(dtype)

        # Convert read_ids to UUIDs
        self.read_ids = [UUID(rid) for rid in read_ids]

        # Initialize POD5 reader
        self.pod5_reader = pod5.DatasetReader(
            pod5_paths, recursive=True, threads=4, max_cached_readers=0
        )

        # Store chunking and batching parameters
        self.window_size = window_size
        self.step_size = step_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.transform = transform

        # Initialize batches
        self._init_batches()

    def _get_signal_dtype(self, dtype: str):
        """
        Returns the corresponding NumPy dtype for signal processing.

        Args:
            dtype (str): Data type specification ('single' or 'double').

        Returns:
            np.dtype: Corresponding NumPy data type.
        """
        if dtype == "float32":
            self.signal_dtype = np.float32
        elif dtype == "float64":
            self.signal_dtype = np.float64
        else:
            raise ValueError("Invalid dtype argument. Use 'float32' or 'float64'.")

    def _init_batches(self):
        """
        Creates batch indices based on batch size for efficient loading.
        """
        self.batches = [
            (i, min(i + self.batch_size, len(self.read_ids)))
            for i in range(0, len(self.read_ids), self.batch_size)
        ]

    def __getitem__(self, index: int):
        """
        Fetches a batch of samples based on the index.

        Args:
            index (int): Index of the batch to retrieve.

        Returns:
            list: List of processed samples in the batch.
        """
        idx_start, idx_end = self.batches[index]
        batch_read_ids = set(self.read_ids[idx_start:idx_end])
        batch_samples = []

        for pod5_record in self.pod5_reader.reads(selection=batch_read_ids):
            try:
                read_id = str(pod5_record.read_id)
                signal = pod5_record.signal.astype(self.signal_dtype)

                if self.transform:
                    signal = self.transform(signal).astype(self.signal_dtype)

                signal = load_signal(
                    signal, 
                    self.window_size, 
                    self.step_size, 
                    self.max_seq_len
                )
                output = {
                    "inputs": {
                        "signal": signal,
                        "length": signal.shape[0],
                    },
                    "metadata": {
                        "read_id": read_id,
                        "num_tokens": signal.shape[0],
                    },
                }

                batch_samples.append(output)
            except Exception as e:
                if self.debug:
                    print(f"Error processing read {read_id}: {e}")
                    raise e
                else:
                    print(f"Error processing read {read_id}. Skipping...")

        return batch_samples

    def __len__(self) -> int:
        """
        Returns the total number of batches in the dataset.

        Returns:
            int: Number of batches available.
        """
        return len(self.batches)
