# stream_inference.py
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import UUID

import numpy as np
import pod5
import torch
import tqdm

from ..config.model_config import ModelConfig, ModelLoader
from ..feeders import SequenceStandardizer, load_signal, collate_fn
from ..storages import InferenceResults, InferenceMetadata
from ..utils import PathSet


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]


class Inference:
    """
    Stream reads from POD5, preprocess on CPU, batch on-the-fly,
    and run inference on GPU – all in a producer/consumer pipeline.
    """

    # ---------- construction -------------------------------------------------
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

    # ---------- public API ---------------------------------------------------
    def predict(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        read_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        save_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
    ) -> InferenceResults:
        """
        Stream-infer over one POD5 file.

        Args
        ----
        pod5_paths   : path to a POD5 file (recursive dirs can be handled outside)
        read_ids    : list of read-ids to restrict to; None ➜ all reads
        batch_size  : how many reads per GPU batch
        save_path   : optional .npz /.json path for results
        show_progress : show tqdm progress bars
        """
        start = time.time()
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)

        metadata = self._build_metadata(pod5_pathset, batch_size)
        results = InferenceResults(metadata=metadata)

        # queue holds *individual* pre-processed read dicts
        sample_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(
            maxsize=batch_size * self.queue_size_mult
        )

        # --- producer thread -------------------------------------------------
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                sample_queue=sample_queue,
                show_progress=show_progress,
            ),
            daemon=True,
            name="pod5-producer",
        )
        producer.start()

        # --- consumer loop (GPU) --------------------------------------------
        self._consumer_loop(
            sample_queue=sample_queue,
            results=results,
            batch_size=batch_size,
        )

        producer.join()
        results.metadata.total_inference_time = time.time() - start

        if save_path:
            results.save(save_path)
            logger.info("Results saved to %s", save_path)

        logger.info(
            "Done – processed %d reads in %.2fs",
            len(results),
            results.metadata.total_inference_time,
        )
        return results

    # ---------- producer / consumer helpers ---------------------------------
    def _producer_worker(
        self,
        *,
        pod5_paths: Path,
        sample_queue: "queue.Queue[Optional[Dict]]",
        read_ids: Optional[List[str]],
        show_progress: bool,
    ) -> None:
        """CPU thread: stream reads → standardise → chunk → queue."""
        try:
            with pod5.DatasetReader(
                pod5_paths, recursive=True, max_cached_readers=4, threads=4
            ) as reader:
                # select reads
                reads_iter = (
                    reader.reads(selection={UUID(r) for r in read_ids})
                    if read_ids
                    else reader.reads()
                )
                if show_progress:
                    reads_iter = tqdm.tqdm(
                        reads_iter,
                        desc="Reading",
                        total=len(read_ids) if read_ids else None,
                        leave=False,
                    )

                for rec in reads_iter:
                    try:
                        sample_queue.put(self._process_record(rec), block=True)
                    except Exception as exc:
                        logger.warning("Failed on read %s: %s", rec.read_id, exc)
        finally:
            # sentinel to tell consumer we are done
            sample_queue.put(None)

    def _consumer_loop(
        self,
        *,
        sample_queue: "queue.Queue[Optional[Dict]]",
        results: InferenceResults,
        batch_size: int,
    ) -> None:
        """Main thread: pop samples, build batches, run GPU, write results."""
        current_batch: List[Dict] = []
        finished = False

        while not finished:
            sample = sample_queue.get()
            if sample is None:
                finished = True
            else:
                current_batch.append(sample)

            if finished or len(current_batch) == batch_size:
                if current_batch:
                    self._run_gpu_batch(current_batch, results)
                    current_batch = []

    # ---------- GPU execution ----------------------------------------------
    def _run_gpu_batch(self, batch: List[Dict], results: InferenceResults) -> None:
        """Collate, push to GPU, run model, store logits."""
        batch_t = collate_fn([batch,])

        inputs = {k: v.to(self.device) for k, v in batch_t["inputs"].items()}
        use_amp = self.config.float_dtype == "float16" and self.device != "cpu"

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device, enabled=use_amp
        ):
            outputs = self.model(**inputs)

        for i, read_id in enumerate(batch_t["metadata"]["read_id"]):
            num_chunks = int(batch_t["metadata"]["num_tokens"][i])
            logits = dict(
                seq_class=outputs["seq_class"][i].cpu().numpy(),
                seq2seq=outputs["seq2seq"][i].cpu().numpy()[:num_chunks],
            )
            results._add(read_id=read_id, logits=logits, num_chunks=num_chunks)

    # ---------- record-level helpers ---------------------------------------
    def _process_record(self, rec: pod5.Record) -> Dict:
        """Return a single pre-processed sample dict for one read."""
        dtype = np.float64 if self.config.float_dtype == "float64" else np.float32
        sig = rec.signal.astype(dtype)
        sig = self._local_standardize(sig)
        sig = load_signal(
            sig,
            window_size=self.config.chunk_size,
            step_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
        )
        sig = sig.astype(dtype)
        return dict(
            inputs=dict(signal=sig, length=sig.shape[0]),
            metadata=dict(read_id=str(rec.read_id), num_tokens=sig.shape[0]),
        )

    @staticmethod
    def _local_standardize(signal: np.ndarray) -> np.ndarray:
        """Per-read z-score normalisation."""
        return SequenceStandardizer().fit_transform(
            [signal.reshape(-1, 1)]
        )[0].ravel()

    # ---------- utilities ---------------------------------------------------
    @staticmethod
    def _load_config(cfg: Union[ModelConfig, str, Dict]) -> ModelConfig:
        if isinstance(cfg, ModelConfig):
            return cfg
        if isinstance(cfg, str):
            if cfg.endswith(".yaml"):
                return ModelConfig.from_yaml(cfg)
            if cfg.endswith(".json"):
                return ModelConfig.from_json(cfg)
            raise ValueError(f"Unsupported config file type: {cfg}")
        if isinstance(cfg, dict):
            return ModelConfig.from_dict(cfg)
        raise TypeError("config must be ModelConfig | str | dict")
    
    def _resolve_paths(self, paths: PathLikeList) -> List[Path]:
        if isinstance(paths, (str, Path)):
            return PathSet([paths])
        elif isinstance(paths, list):
            return PathSet(paths)
        else:
            raise TypeError(f"Expected str, Path, or list of them, got {type(paths).__name__}")

    def _build_metadata(self, pod5_pathset: PathSet, batch_size: int) -> InferenceMetadata:
        return InferenceMetadata(
            chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            model_type=getattr(self.config, "model_type", "transformer"),
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            num_classes_seq2seq=getattr(self.config, "num_classes_seq2seq", 4),
            label_names=self.config.label_names,
            batch_size=batch_size,
            device=str(self.device),
            float_dtype=self.config.float_dtype,
            model_checkpoint=str(getattr(self.config, "checkpoint_path", None)),
            pod5_paths=pod5_pathset.to_list(),
        )

    # ---------- context manager --------------------------------------------
    def __enter__(self) -> "Inference":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
