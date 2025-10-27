from typing import Optional, Sequence, Union, List, Tuple, TYPE_CHECKING
import numpy as np
import torch
from torch import nn
from torchcrf import CRF

if TYPE_CHECKING:
    from ..storages import ReadResult, ReadResultCompressed

class CRFSmoother:
    def __init__(
        self,
        num_tags: int = 4,
        transitions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_tags = num_tags
        self.device = torch.device(device)
        self.dtype = dtype

        self.crf = CRF(num_tags=num_tags, batch_first=True).to(self.device)

        # transitions
        if transitions is None:
            if num_tags != 4:
                raise ValueError("Provide transitions for num_tags != 4.")
            trans = torch.tensor(
                [[0, -1e8, -1e8, 0],
                 [-1e8, 0, 0, -1e8],
                 [0, -1e8, 0, -1e8],
                 [-1e8, -1e8, -1e8, 0]], dtype=dtype, device=self.device
            )
        else:
            trans = torch.as_tensor(transitions, dtype=dtype, device=self.device)
            if trans.shape != (num_tags, num_tags):
                raise ValueError(f"transitions must be {(num_tags, num_tags)}, got {tuple(trans.shape)}")

        self.crf.transitions = nn.Parameter(trans)

        # neutral start/end unless you have a reason otherwise
        self.crf.start_transitions.data.zero_()
        self.crf.end_transitions.data.zero_()

    def to(self, device: Union[torch.device, str]) -> "CRFSmoother":
        self.device = torch.device(device)
        self.crf.to(self.device)
        return self

    @staticmethod
    def _ensure_batch3d(
        logits: Union[np.ndarray, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        em = torch.as_tensor(logits, dtype=dtype, device=device)
        if em.ndim == 2:
            em = em.unsqueeze(0)
            single = True
        elif em.ndim == 3:
            single = False
        else:
            raise ValueError(f"logits must be (B,T,C) or (T,C); got {tuple(em.shape)}")
        return em, single

    @staticmethod
    def _build_mask(B: int, T: int, lengths: Optional[Sequence[int]], device: torch.device) -> torch.Tensor:
        if lengths is None:
            return torch.ones((B, T), dtype=torch.bool, device=device)
        if len(lengths) != B:
            raise ValueError(f"lengths must have size {B}, got {len(lengths)}")
        lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=device)
        return torch.arange(T, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)

    def decode(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        lengths: Optional[Sequence[int]] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray], List[List[int]]]:
        emissions, single = self._ensure_batch3d(logits, self.dtype, self.device)
        B, T, C = emissions.shape
        if C != self.num_tags:
            raise ValueError(f"Expected C={self.num_tags}, got C={C}")

        mask = self._build_mask(B, T, lengths, self.device)

        with torch.inference_mode():
            paths: List[List[int]] = self.crf.decode(emissions, mask)

        if return_numpy:
            paths_np = [np.asarray(p, dtype=np.int64) for p in paths]
            return paths_np[0] if single else paths_np
        else:
            return paths[0] if single else paths
        
    def decode_read_results(
        self,
        read_results: List["ReadResult"],
        return_numpy: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray], List[List[int]]]:
        if not read_results:
            raise ValueError("read_results is empty")

        from ..storages import ReadResult, ReadResultCompressed
        # Collect per-read logits as tensors (Ti, C)
        seqs = []
        lens = []
        C_expected = self.num_tags
        for rr in read_results:
            if isinstance(rr, ReadResultCompressed):
                raise NotImplementedError("CRF smoothing can only be applied to raw ReadResult, not ReadResultCompressed")
            em = torch.as_tensor(rr.segmentation_logits, dtype=self.dtype, device=self.device)
            if em.ndim != 2:
                raise ValueError(f"Each segmentation_logits must be (T,C); got {tuple(em.shape)}")
            Ti, C = em.shape
            if C != C_expected:
                raise ValueError(f"num_tags mismatch: expected C={C_expected}, got C={C}")
            # If rr.num_chunks is provided, trust it (and ensure it doesn't exceed Ti)
            L = int(rr.num_chunks) if hasattr(rr, "num_chunks") and rr.num_chunks is not None else Ti
            if L < 0 or L > Ti:
                raise ValueError(f"num_chunks ({L}) out of bounds for T={Ti}")
            seqs.append(em)
            lens.append(L)

        B = len(seqs)
        max_T = max(lens) if lens else 0
        if max_T == 0:
            # all zero-length; return empty paths
            return [[] for _ in range(B)]  # or np.empty(0, dtype=np.int64) per item

        # Pad to (B, max_T, C)
        emissions = torch.zeros((B, max_T, C_expected), dtype=self.dtype, device=self.device)
        for i, (em, L) in enumerate(zip(seqs, lens)):
            emissions[i, :L, :] = em[:L, :]

        # Build mask from lengths (left-justified)
        lengths = torch.tensor(lens, dtype=torch.long, device=self.device)
        mask = torch.arange(max_T, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_T)
        # CRF expects uint8/bool is fine
        with torch.inference_mode():
            paths: List[List[int]] = self.crf.decode(emissions, mask)

        if return_numpy:
            return [np.asarray(p, dtype=np.int64) for p in paths]
        else:
            return paths
