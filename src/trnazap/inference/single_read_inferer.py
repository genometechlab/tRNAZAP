from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
from uuid import UUID
from dataclasses import dataclass, field

import numpy as np
import pod5
import torch
import tqdm

from .inference_base import InferenceBase
from ..config.model_config import ModelConfig, ModelLoader
from ..feeders import SequenceStandardizer, collate_fn
from ..storages import InferenceResults, InferenceMetadata, ReadResultDetailed
from ..utils import PathSet
from ..io import ZIRWriter, ZIRShardManager


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

class SingleReadInference(InferenceBase):
    """
    Run inference on one read from POD5, optionally returning CLS attention
    and token saliency for interpretability.
    """

    def predict(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        read_id: str,
        return_attention_scores: bool = False,
        return_token_saliency: bool = False,
        saliency_target_class: Optional[int] = None,
        saliency_use_abs: Optional[bool] = False,
        saliency_reduce: Optional[str] = 'l2',
        saliency_normalize: Optional[bool] = False,
    ) -> Union[ReadResultDetailed, Tuple[ReadResultDetailed, Dict]]:
        """
        Returns the read's ReadResultDetailed, or ``(result, aux)`` when attention
        or saliency is requested. Raises KeyError if the read is not in the POD5 files.
        """
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        read_id = str(UUID(read_id))  # validate and normalise
        model_input = None
        with pod5.DatasetReader(
            pod5_pathset.paths, recursive=True, max_cached_readers=4, threads=4
        ) as reader:
            for rec in reader.reads(selection={UUID(read_id)}):
                try:
                    model_input = self._process_record(rec)
                except Exception as exc:
                    raise RuntimeError(f"Failed to preprocess read {read_id}") from exc

        if model_input is None:
            raise KeyError(f"Read {read_id} not found in {pod5_pathset.to_list()}")

        batch_t = collate_fn([model_input,])

        inputs = {k: v.to(self.device) for k, v in batch_t["inputs"].items()}
        use_amp = (self.config.float_dtype in {"float16", "fp16"}) and (self.device.type != "cpu")
        dtype = torch.float16 if self.config.float_dtype in {"float16", "fp16"} else None

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=use_amp, dtype=dtype
        ):
            outputs = self.model(**inputs)

        num_chunks = int(batch_t["metadata"]["num_tokens"][0])
        cropped = bool(batch_t["metadata"]["cropped"][0])
        logits = {
            k: v[0].cpu().numpy()
            for k, v in outputs.items()
        }
        
        result = ReadResultDetailed(
            read_id=read_id,
            _logits=logits,
            num_chunks=num_chunks,
            chunk_size=self.config.chunk_size,
            cropped=cropped
        )
        
        aux_out = dict()
        
        if return_attention_scores and hasattr(self.model, 'get_cls_attention'):
            cls_attn, cls_attn_mean = self.model.get_cls_attention(**inputs, average_heads=True)
            cls_attn = cls_attn.cpu().numpy()[0]
            cls_attn_mean = cls_attn_mean.cpu().numpy()[0]
            aux_out["attention"] = (cls_attn, cls_attn_mean)
            
        if return_token_saliency and hasattr(self.model, 'get_token_saliency'):
            sal, chosen = self.model.get_token_saliency(**inputs, 
                                                        target_class=saliency_target_class, 
                                                        use_abs = saliency_use_abs,
                                                        reduce=saliency_reduce,
                                                        normalize=saliency_normalize)
            sal = sal.cpu().numpy()[0]
            chosen = chosen.cpu().numpy()[0]
            aux_out["saliency"] = (sal, chosen)
            
        if aux_out: return result, aux_out
        else: return result