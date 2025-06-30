import numpy as np
import pickle


class SequenceScaler:
    def __init__(self, scale=2, offset=-1):
        self.scale = scale
        self.offset = offset
        self.min_array = None
        self.max_array = None

    def fit(self, signal_arrays):
        if isinstance(signal_arrays, np.ndarray):
            if signal_arrays.ndim == 1:
                signal_arrays = signal_arrays.reshape(-1, 1)
            signal_arrays = [signal_arrays]

        min_array = np.full((signal_arrays[0].shape[1],), np.inf)
        max_array = np.full((signal_arrays[0].shape[1],), -np.inf)

        for signal_array in signal_arrays:
            min_array = np.minimum(min_array, np.nanmin(signal_array, axis=0))
            max_array = np.maximum(max_array, np.nanmax(signal_array, axis=0))

        self.min_array = min_array.reshape(1, -1)
        self.max_array = max_array.reshape(1, -1)

        self.scale_factor = self.scale / (self.max_array - self.min_array + 1e-8)
        self.offset_factor = self.offset - self.min_array * self.scale_factor

    def transform(self, signal_arrays):
        if self.min_array is None or self.max_array is None:
            raise ValueError(
                "SequenceScaler has not been fitted yet. Call fit() first."
            )

        transformed_arrays = []
        for signal_array in signal_arrays:
            scaled_array = np.nan_to_num(
                signal_array * self.scale_factor + self.offset_factor
            )
            transformed_arrays.append(scaled_array)

        return transformed_arrays

    def fit_transform(self, signal_arrays):
        self.fit(signal_arrays)
        return self.transform(signal_arrays)

    def get_min_max(self):
        return self.min_array, self.max_array

    def set_min_max(self, pth):
        with open(pth, "rb") as f:
            min_array, max_array = pickle.load(f)
        self.min_array = min_array
        self.max_array = max_array
        self.scale_factor = self.scale / (self.max_array - self.min_array + 1e-8)
        self.offset_factor = self.offset - self.min_array * self.scale_factor
