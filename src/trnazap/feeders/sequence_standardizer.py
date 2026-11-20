import numpy as np
import pickle


class SequenceStandardizer:
    def __init__(self):
        self.mean_array = None
        self.std_array = None

    def fit(self, signal_arrays):
        # Initialize accumulators for mean and variance calculation
        total_sum = np.zeros((signal_arrays[0].shape[1],))
        total_sq_sum = np.zeros((signal_arrays[0].shape[1],))
        total_count = 0

        # Compute the mean and variance across all signal arrays
        for signal_array in signal_arrays:
            total_sum += np.nansum(signal_array, axis=0)
            total_sq_sum += np.nansum(signal_array**2, axis=0)
            total_count += signal_array.shape[0]

        # Calculate mean and standard deviation
        self.mean_array = total_sum / total_count
        variance = (total_sq_sum / total_count) - (self.mean_array**2)
        # Add a small value to avoid division by zero
        self.std_array = np.sqrt(variance + 1e-8)

        # Reshape for broadcasting
        self.mean_array = self.mean_array.reshape(1, -1)
        self.std_array = self.std_array.reshape(1, -1)

    def transform(self, signal_arrays):
        if self.mean_array is None or self.std_array is None:
            raise ValueError(
                "SequenceStandardizer has not been fitted yet. Call fit() first."
            )

        transformed_arrays = []
        for signal_array in signal_arrays:
            standardized_array = (signal_array - self.mean_array) / self.std_array
            transformed_arrays.append(
                np.nan_to_num(standardized_array)
            )

        return transformed_arrays

    def fit_transform(self, signal_arrays):
        self.fit(signal_arrays)
        return self.transform(signal_arrays)

    def get_mean_std(self):
        return self.mean_array, self.std_array

    def set_mean_std(self, pth):
        with open(pth, "rb") as f:
            mean, std = pickle.load(f)
        self.mean_array = mean
        self.std_array = std


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    """Per-read z-score of a 1-D signal, returned as float64.

    Gives exactly the same numbers as
    ``SequenceStandardizer().fit_transform([signal.reshape(-1, 1)])[0].ravel()``
    (same dtypes, formula and order of operations), without its per-call
    overhead: no list/reshape wrapping and no NaN handling, which does nothing
    for raw nanopore signal (integers, so never NaN or inf). The rare cases
    where that handling matters (empty signal, variance rounding below zero)
    fall back to the same ``nan_to_num`` result. Bit-identity is checked by
    tests/test_standardize_signal.py.
    """
    n = signal.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.float64(signal.sum()) / n
        mean_sq = np.float64(np.square(signal).sum()) / n
        # mean * mean, not mean ** 2: the old code squared a float64 *array*
        # (x * x); a scalar ** 2 can differ from it in the last bit.
        std = np.sqrt(mean_sq - mean * mean + 1e-8)
        # Explicit float64, as the old (1, 1)-array arithmetic gave on any numpy
        # version (a float64 *scalar* would keep float32 on numpy < 2).
        out = np.subtract(signal, mean, dtype=np.float64)
        out /= std
    if not (np.isfinite(mean) and np.isfinite(std)):
        out = np.nan_to_num(out)
    return out
