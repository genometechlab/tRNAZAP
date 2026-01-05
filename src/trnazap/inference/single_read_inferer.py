from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator
from uuid import UUID
from dataclasses import dataclass, field

import numpy as np
import pod5
import torch
import tqdm

from .inference_base import InferenceBase
from ..config.model_config import ModelConfig, ModelLoader
from ..feeders import SequenceStandardizer, load_signal, collate_fn
from ..storages import InferenceResults, InferenceMetadata, ReadResult
from ..utils import PathSet
from ..io import ZIRWriter, ZIRShardManager


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

class SingleReadInference(InferenceBase):
    """
    Stream reads from POD5, preprocess on CPU, batch on-the-fly,
    and run inference on GPU.
    """

    def __init__(
        self,
        config: Union[ModelConfig, str, Dict],
        device: Optional[torch.device] = None,
    ) -> None:
        self.config: ModelConfig = self._load_config(config)
        self.device: torch.device = (
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.queue_size_mult = 4
        
        self.model_loader = ModelLoader(self.config, self.device)
        self.model = self.model_loader.get_model(load_checkpoint=True).eval()

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
    ) -> Optional[InferenceResults]:
        
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        with pod5.DatasetReader(
            pod5_pathset.paths, recursive=True, max_cached_readers=4, threads=4
        ) as reader:
            reads_iter = (reader.reads(selection={UUID(r) for r in [read_id,]}))
            for rec in reads_iter:
                try:
                    model_input = self._process_record(rec)
                except Exception as exc:
                    logger.warning("Failed on read %s: %s", rec.read_id, exc)

        batch_t = collate_fn([model_input,])
        
        assert batch_t["metadata"]["read_id"][0] == read_id

        inputs = {k: v.to(self.device) for k, v in batch_t["inputs"].items()}
        use_amp = self.config.float_dtype == "float16" and self.device != "cpu"

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device, enabled=use_amp
        ):
            outputs = self.model(**inputs)

        num_chunks = int(batch_t["metadata"]["num_tokens"][0])
        logits = {
            k: v[0].cpu().numpy()
            for k, v in outputs.items()
        }
        
        result = ReadResult(
            read_id=read_id,
            _logits=logits,
            num_chunks=num_chunks,
            chunk_size=self.config.chunk_size
        )
        
        aux_out = dict()
        
        if return_attention_scores and hasattr(self.model, 'get_cls_attention'):
            cls_scores, cls_attn, cls_attn_mean = self.model.get_cls_attention(**inputs, average_heads=True)
            cls_scores = cls_scores.cpu().numpy()[0]
            cls_attn = cls_attn.cpu().numpy()[0]
            cls_attn_mean = cls_attn_mean.cpu().numpy()[0]
            aux_out["attention"] = (cls_scores, cls_attn, cls_attn_mean)
            
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
        else: result