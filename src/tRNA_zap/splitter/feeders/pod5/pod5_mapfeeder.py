import torch.utils.data as data
import numpy as np
import pod5
import torch
from uuid import UUID
from .pod5_utils import load_signal, load_signal_with_tokens


@DeprecationWarning
class Pod5MapDataset(data.Dataset):
    def __init__(
        self,
        read_ids: list,
        pod5_paths: str,
        window_size: int,
        step_size: int,
        load_labels: bool = True,
        token_rles: list = None,
        labels: list = None,
        rle_decoder: callable = None,
        rle_decoder_args: dict = None,
        transform: callable = None,
        random_crop: bool = False,
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
            load_labels (bool): Whether to load labels and token run-length encodings (RLE). Defaults to True.
            token_rles (list, optional): Encoded token run-lengths, required if load_labels is True.
            labels (list, optional): Corresponding labels for reads, required if load_labels is True.
            rle_decoder (callable, optional): Function to decode RLE tokens, required if load_labels is True.
            rle_decoder_args (dict, optional): Arguments for the RLE decoder, required if load_labels is True.
            transform (callable, optional): Transformation function for signals.
            random_crop (bool, optional): Whether to apply random cropping. Defaults to False.
            dtype (str, optional): Data type of the signal ('single' or 'double'). Defaults to 'single'.
            debug (bool, optional): If True, enables debug mode with error logging. Defaults to False.
        """
        
        # Set signal data type
        self._get_signal_dtype(dtype)

        # assign read_ids
        self.read_ids = read_ids

        # Initialize POD5 reader
        self.pod5_reader = pod5.DatasetReader(pod5_paths, index=True, threads=2)

        # Store chunking and batching parameters
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.random_crop = random_crop
        self.debug = debug
        self.load_labels = load_labels

        self.token_rles = token_rles
        self.labels = labels
        self.rle_decoder = rle_decoder
        self.rle_decoder_args = rle_decoder_args or {}
        self.load_labels = load_labels
        self.debug = debug

        if self.load_labels:
            if len(self.read_ids) != len(self.labels) or len(self.read_ids) != len(
                self.token_rles
            ):
                raise ValueError(
                    "Length of read_ids, token_rles, and labels must be equal."
                )

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

    def __getitem__(self, index):
        try:
            read_id = self.read_ids[index]
            pod5_record = self.pod5_reader.get_read(read_id)
            signal = pod5_record.signal
            signal = signal.astype(self.signal_dtype)
            if self.transform is not None:
                signal = self.transform(signal)
            signal = signal.astype(self.signal_dtype)

            if self.load_labels:
                label = self.labels[index]
                token_rle = self.token_rles[index]
                tokens = self.rle_decoder(token_rle, **self.rle_decoder_args)
                return load_signal_with_tokens(
                    signal,
                    tokens,
                    label,
                    self.window_size,
                    self.step_size,
                    self.random_crop,
                )
            else:
                return load_signal(signal, self.window_size, self.step_size)
        except Exception as e:
            if self.debug:
                raise e
            else:
                print(f"Error loading data at index {index}: {e}")
                return self.__getitem__(index + 1)

    def __len__(self):
        return (len(self.read_ids))
