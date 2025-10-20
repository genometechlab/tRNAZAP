import torch
import numpy as np
from torchcrf import CRF  # Make sure this is installed
from typing import Optional, Sequence, Union, List

@DeprecationWarning
def crf_smoothing_depracated(logits: np.ndarray, device: Union[torch.device, str] = "cpu") -> np.ndarray:
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


def crf_smoothing(
    logits: np.ndarray,
    lengths: Optional[Sequence[int]] = None,
    device: Union[torch.device, str] = "cpu",
) -> Union[np.ndarray, List[np.ndarray]]:
    single = (logits.ndim == 2)  # (T, C) case
    em = torch.as_tensor(logits, dtype=torch.float32)
    if single:
        em = em.unsqueeze(0)  # -> (1, T, C)

    if em.ndim != 3:
        raise ValueError(f"logits must be (B,T,C) or (T,C); got shape {tuple(em.shape)}")

    B, T, C = em.shape

    # This mirrors your original transitions (expects C == 4)
    if C != 4:
        raise ValueError(f"Transition template expects C=4, got C={C}. Provide a custom matrix if needed.")

    crf = CRF(num_tags=C, batch_first=True)
    crf.transitions = torch.nn.Parameter(torch.tensor([
        [0,    -1e8, -1e8,    0],
        [-1e8,    0,    0, -1e8],
        [0,    -1e8,    0, -1e8],
        [-1e8, -1e8, -1e8,    0],
    ], dtype=torch.float32))
    crf.to(device)

    em = em.to(device)

    if lengths is not None:
        if len(lengths) != B:
            raise ValueError(f"lengths must have size {B}, got {len(lengths)}")
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
        mask = (torch.arange(T, device=device).unsqueeze(0) < lengths_t.unsqueeze(1))
    else:
        mask = torch.ones((B, T), dtype=torch.bool, device=device)

    with torch.no_grad():
        decoded = crf.decode(em, mask)  # List[List[int]] length B

    # Convert to numpy
    decoded_np = [np.asarray(path, dtype=np.int64) for path in decoded]

    # Match original single-sample return shape
    if single:
        return decoded_np[0]
    return decoded_np
