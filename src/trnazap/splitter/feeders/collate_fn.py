
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Alternative implementation using pad_sequence first, then adjusting to fixed length.
    This might be cleaner if you have many sequences much shorter than max_seq_len.
    """
    assert len(batch) == 1
    batch = batch[0]
    
    signals = [torch.tensor(item["inputs"]["signal"]) for item in batch]
    lengths = torch.tensor([item["inputs"]["length"] for item in batch])
    read_ids = [item["metadata"]['read_id'] for item in batch]
    
    # First pad to the batch max (which might be less than max_seq_len)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=-1)
    
    return {
        "inputs": {
            "signal": padded_signals,  # Always [batch_size, max_seq_len, input_dim]
            "length": lengths,
        },
        "metadata": {
            "read_id": read_ids,
            "num_tokens": lengths,
        }
    }