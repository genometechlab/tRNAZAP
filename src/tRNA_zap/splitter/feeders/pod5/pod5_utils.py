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

def load_signal_with_tokens(signal, tokens, window_size, step_size, max_seq_len, random_crop=False):
    signal_len = signal.shape[0]

    signal_len = len(signal)
    tokens_len = len(tokens)
    if signal_len < tokens_len:
        tokens = tokens[:signal_len]
    elif signal_len > tokens_len:
        tokens = np.pad(tokens, (0, signal_len - tokens_len), mode="edge")

    if random_crop:
        signal, tokens = apply_random_crop(signal, tokens)

    signal = signal[: (signal_len // window_size) * window_size]
    tokens = tokens[: (signal_len // window_size) * window_size]

    signal = apply_rolling_window(window_size, step_size, signal)
    tokens = apply_rolling_window(window_size, step_size, tokens)

    if max_seq_len and len(signal)>max_seq_len:
        signal = signal[:max_seq_len]
        tokens = tokens[:max_seq_len]
        
    tokens = np.min(tokens, axis=1)
    return signal, tokens

    
def load_signal(signal, window_size, step_size, max_seq_len):

    signal_len = signal.shape[0]
    signal = signal[: (signal_len // window_size) * window_size]
    signal = apply_rolling_window(window_size, step_size, signal)
    if max_seq_len and len(signal)>max_seq_len:
        signal = signal[:max_seq_len]
    return signal

def load_tokens(tokens, window_size, step_size):
    tokens_len = len(tokens)
    tokens = tokens[: (tokens_len // window_size) * window_size]
    tokens = apply_rolling_window(window_size, step_size, tokens)
    tokens = np.min(tokens, axis=1)
    return tokens
