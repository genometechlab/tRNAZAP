import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate a list of sample dicts into a batched dict.

    Each sample carries:
        inputs.signal : np.ndarray [N]   raw 1-D signal (z-scored, truncated)
        inputs.length : int              raw sample count (not token count)
        metadata.*

    Returns
    -------
    {
        "inputs": {
            "signal": FloatTensor [B, N_max],   padded with 0.0
            "length": LongTensor  [B],           raw sample counts
        },
        "metadata": {
            "read_id":    List[str]
            "num_tokens": LongTensor [B]
        }
    }
    """
    signals  = [torch.tensor(item["inputs"]["signal"]) for item in batch]
    lengths  = torch.tensor([item["inputs"]["length"]  for item in batch], dtype=torch.long)
    num_toks = torch.tensor([item["metadata"]["num_tokens"] for item in batch], dtype=torch.long)
    read_ids = [item["metadata"]["read_id"] for item in batch]

    # Pad 1-D signals to batch-max length with 0.0
    # (padding value 0.0 is safe — z-scored signal has zero mean,
    #  and the model masks padded positions anyway)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=0.0)  # [B, N_max]

    return {
        "inputs": {
            "signal": padded_signals,   # [B, N_max]
            "length": lengths,          # [B] raw sample counts
        },
        "metadata": {
            "read_id":    read_ids,
            "num_tokens": num_toks,
        },
    }