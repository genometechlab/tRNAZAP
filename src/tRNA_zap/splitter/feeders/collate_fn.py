
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from functools import partial

def collate_fn(batch, max_seq_len=1000):
    """
    Alternative implementation using pad_sequence first, then adjusting to fixed length.
    This might be cleaner if you have many sequences much shorter than max_seq_len.
    """
    assert len(batch) == 1
    batch = batch[0]
    
    signals = [torch.tensor(item["inputs"]["signal"])[:max_seq_len] for item in batch]
    lengths = torch.tensor([min(item["inputs"]["length"], max_seq_len) for item in batch])
    read_ids = [item["metadata"]['read_id'] for item in batch]
    
    # First pad to the batch max (which might be less than max_seq_len)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=-1)
    
    # Then pad to max_seq_len if needed
    current_max_len = padded_signals.size(1)
    if current_max_len < max_seq_len:
        pad_len = max_seq_len - current_max_len
        padded_signals = F.pad(padded_signals, (0, 0, 0, pad_len), value=-1)
    
    return {
        "inputs": {
            "signal": padded_signals,  # Always [batch_size, max_seq_len, input_dim]
            "length": lengths,
        },
        "metadata": {
            "read_id": read_ids,
            "num_tokens": lengths,
            "max_seq_len": max_seq_len,
        }
    }

def collate_fn_fixed_padding(max_seq_len=1000):
    """
    Factory function to create a collate_fn with a specific max_seq_len.
    
    Usage:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=create_fixed_padding_collate_fn(max_seq_len=2000)
        )
    """
    return partial(collate_fn, max_seq_len=max_seq_len)