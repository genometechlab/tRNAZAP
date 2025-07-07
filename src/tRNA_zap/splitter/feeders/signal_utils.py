import numpy as np

def apply_rolling_window(window_size, step_size, array):
    if array.shape[0] < window_size:
        raise ValueError("Array size must be greater than window size.")
    shape = (array.size - window_size + 1, window_size)
    strides = array.strides * 2
    windows = np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    )[0::step_size]
    return windows

def apply_random_crop(signal, tokens):
    start_indices = np.where((tokens ==1 ) | (tokens == 2))[0]
    end_indices = np.where(tokens == 3)[0]

    start = np.random.choice(start_indices) if len(start_indices)!=0 else 0
    end = np.random.choice(end_indices) if len(end_indices)!=0 else len(tokens)

    return signal[start:end], tokens[start:end]

def load_signal(signal, window_size, step_size, max_seq_len):

    signal_len = signal.shape[0]
    signal = signal[: (signal_len // window_size) * window_size]
    signal = apply_rolling_window(window_size, step_size, signal)
    if max_seq_len and len(signal)>max_seq_len:
        signal = signal[:max_seq_len]
    return signal
