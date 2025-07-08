import torch
import numpy as np
from torchcrf import CRF  # Make sure this is installed
from typing import Union

def crf_smoothing(logits: np.ndarray, device: Union[torch.device, str] = "cpu") -> np.ndarray:
    crf = CRF(4, batch_first=True)
    crf.transitions = torch.nn.Parameter(torch.tensor([
        [0, -1e8, -1e8, 0],
        [-1e8, 0, 0, -1e8],
        [0, -1e8, 0, -1e8],
        [-1e8, -1e8, -1e8, 0],
    ]))
    crf.to(device)
    with torch.no_grad():
        logits_tensor = torch.tensor(logits).unsqueeze(0).to(device)
        mask = torch.ones(logits_tensor.shape[:2], dtype=torch.bool).to(device)
        decoded = crf.decode(logits_tensor, mask)
    return np.array(decoded[0])
