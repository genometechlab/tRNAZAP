from __future__ import annotations

from abc import ABC, abstractmethod

import hashlib
import importlib.metadata
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pod5
import torch
from uuid import UUID

from ..config.model_config import ModelConfig, ModelLoader, config_file
from ..feeders import SequenceStandardizer
from ..storages import InferenceResults, InferenceMetadata, ReadResultDetailed
from ..utils import PathSet
from ..io import ZIRShardManager


logger = logging.getLogger(__name__)


def _sha256(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _trnazap_version() -> Optional[str]:
    try:
        return importlib.metadata.version("trnazap")
    except importlib.metadata.PackageNotFoundError:
        return None

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

class InferenceBase(ABC):

    def __init__(
        self,
        config: Union[ModelConfig, str, Dict],
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: ModelConfig = ModelConfig.load_config(cfg=config)
        # File name only: paths are machine-specific; the weights are pinned by hash
        self.config_name = config_file(config).name if isinstance(config, str) else None
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_size_mult = 4
        self.model_loader = ModelLoader(self.config, self.device)
        self.model = self.model_loader.get_model(load_checkpoint=True).eval()
        self.checkpoint_sha256 = _sha256(self.model_loader.checkpoint_path)

    def _process_record(self, rec: pod5.ReadRecord) -> Dict:
        """
        Return a single pre-processed sample dict for one read.

        Signal is z-scored and truncated to the nearest multiple of
        effective_stride, then stored as a 1-D array [N].
        Token count is computed here so collate_fn and the model
        receive consistent length values.
        """
        dtype = np.float64 if self.config.float_dtype == "float64" else np.float32

        sig = rec.signal.astype(dtype)
        sig = self._local_standardize(sig)

        # Truncate to nearest multiple of effective_stride
        stride = self.config.effective_stride      # replaces chunk_size in pipeline
        max_samples = self.config.max_seq_len * stride
        sig = sig[: (sig.shape[0] // stride) * stride]  # drop last incomplete chunk
        cropped = False
        if len(sig) > max_samples:
            sig = sig[:max_samples]
            cropped = True

        num_tokens = len(sig) // stride
        sig = sig.astype(dtype)

        return dict(
            inputs=dict(
                signal=sig,           # [N]  raw 1-D signal
                length=len(sig),      # raw sample count — model converts to tokens
            ),
            metadata=dict(
                read_id=str(rec.read_id),
                num_tokens=num_tokens,
                cropped=cropped
            ),
        )

    @staticmethod
    def _local_standardize(signal: np.ndarray) -> np.ndarray:
        """Per-read z-score normalisation."""
        return SequenceStandardizer().fit_transform(
            [signal.reshape(-1, 1)]
        )[0].ravel()

    def _resolve_paths(self, paths: PathLikeList) -> PathSet:
        if isinstance(paths, (str, Path)):
            return PathSet([paths])
        elif isinstance(paths, list):
            return PathSet(paths)
        else:
            raise TypeError(
                f"Expected str, Path, or list of them, got {type(paths).__name__}"
            )

    def _build_metadata(
        self, pod5_pathset: PathSet, batch_size: int, run_id: Optional[str] = None
    ) -> InferenceMetadata:
        return InferenceMetadata(
            chunk_size=self.config.effective_stride,
            max_seq_len=self.config.max_seq_len,
            model_name=self.config.model_name,
            num_classification_classes=self.config.num_classification_classes,
            num_segmentation_classes=getattr(self.config, "num_segmentation_classes", 4),
            label_names=self.config.label_names,
            batch_size=batch_size,
            device=str(self.device),
            float_dtype=self.config.float_dtype,
            pod5_paths=pod5_pathset.to_list(),
            config_name=self.config_name,
            tasks=sorted(self.model.enabled_tasks),
            head_type=self.config.head_type,
            checkpoint_sha256=self.checkpoint_sha256,
            run_id=run_id,
            trnazap_version=_trnazap_version(),
        )

    def __enter__(self) -> "InferenceBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass