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
from tqdm.auto import tqdm

from .inference_base import InferenceBase
from ..config.model_config import ModelConfig, ModelLoader
from ..feeders import SequenceStandardizer, load_signal, collate_fn
from ..storages import InferenceResults, InferenceMetadata, ReadResult, ReadResultCompressed
from ..utils import PathSet
from ..utils import crf_smoothing
from ..io import ZIRWriter, ZIRShardManager


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

class Inference(InferenceBase):
    """
    Stream reads from POD5, preprocess on CPU, batch on-the-fly,
    and run inference on GPU.
    """

    def __init__(
        self,
        config: Union[ModelConfig, str, Dict],
        device: Optional[torch.device] = None,
        save_raw: bool = True,
    ) -> None:
        self.config: ModelConfig = self._load_config(config)
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.save_raw = save_raw

        self.queue_size_mult = 4
        
        self.model_loader = ModelLoader(self.config, self.device)
        self.model = self.model_loader.get_model(load_checkpoint=True).eval()

    def predict(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        output_path: Optional[Union[str, Path]] = None,
        read_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        shard_size: Optional[int] = None,
        show_progress: bool = True,
        return_results: bool = True,
    ) -> Optional[InferenceResults]:
        """
        Run inference on POD5 files.

        Args
        ----
        pod5_paths    : Path to POD5 file(s) or directory
        output_path   : Optional path to save ZIR file(s). If None, only return in-memory results
        read_ids      : List of read-ids to process; None = all reads
        batch_size    : Number of reads per GPU batch
        shard_size    : If saving to disk, number of reads per shard (None = single file)
        show_progress : Show progress bars
        return_results: If True, return InferenceResults object (keeps all in memory)
                       If False, only write to disk (memory efficient for large datasets)
        
        Returns
        -------
        InferenceResults if return_results=True, else None
        
        Examples
        --------
        # Notebook/interactive use - get results in memory
        results = inference.predict("data.pod5")
        print(f"Processed {len(results)} reads")
        
        # Large dataset - write directly to disk
        inference.predict("large_data.pod5", output_path="output.zir", return_results=False)
        
        # Both - useful for moderate datasets
        results = inference.predict("data.pod5", output_path="output.zir")
        """
        if not return_results and output_path is None:
            raise ValueError("Nothing to do: set output_path or return_results=True")
                     
        start = time.time()
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        
        num_reads = self._get_num_reads(read_ids, pod5_pathset)

        metadata = self._build_metadata(pod5_pathset, batch_size)
        
        # Setup storage strategy
        in_memory_results = None
        zir_manager = None
        
        if return_results:
            in_memory_results = InferenceResults(metadata)
            
        if output_path:
            zir_manager = ZIRShardManager(
                base_path=Path(output_path),
                metadata=metadata,
                shard_size=shard_size,
            )
            
        
        producer_pbar = None
        consumer_pbar = None
        if show_progress:
            bar_fmt = (
                "{desc:<20} "
                "{percentage:6.2f}%|"
                "{bar:30}| "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}/{remaining}, {rate_fmt}]"
            )
            producer_pbar = tqdm(
                desc="Reading POD5",
                unit="reads",
                total=num_reads,
                leave=True,
                bar_format=bar_fmt,
                position=0
            )
            consumer_pbar = tqdm(
                desc="Processed reads",
                unit="reads",
                total=num_reads,
                leave=True,
                bar_format=bar_fmt,
                position=1
            )

        # Queue for preprocessed samples
        sample_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(
            maxsize=batch_size * self.queue_size_mult
        )

        # Producer thread
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                sample_queue=sample_queue,
                progress_bar=producer_pbar
            ),
            daemon=True,
            name="pod5-producer",
        )
        producer.start()

        # Consumer loop
        processed_count = self._consumer_loop(
            sample_queue=sample_queue,
            in_memory_results=in_memory_results,
            zir_manager=zir_manager,
            batch_size=batch_size,
            progress_bar=consumer_pbar
        )

        producer.join()
        
        # Finalize
        total_time = time.time() - start
        
        if zir_manager:
            zir_manager.close()
            
        if in_memory_results:
            in_memory_results.metadata.total_inference_time = total_time
            
        logger.info(
            "Done – processed %d reads in %.2fs",
            processed_count,
            total_time,
        )
        
        return in_memory_results

    def predict_iter(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        read_ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Iterator[ReadResult]:
        """
        Iterator version for streaming results without storing in memory.
        
        Yields
        ------
        ReadResult objects as they are processed
        
        Example
        -------
        for result in inference.predict_iter("data.pod5"):
            if result.get_preds('classification') == 0:
                print(f"Found target read: {result.read_id}")
        """
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        
        # Queue for preprocessed samples
        sample_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(
            maxsize=batch_size * self.queue_size_mult
        )
        
        # Queue for results
        result_queue: "queue.Queue[Optional[ReadResult]]" = queue.Queue(
            maxsize=batch_size * 2
        )

        # Producer thread
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                sample_queue=sample_queue,
                progress_bar=None,
            ),
            daemon=True,
        )
        producer.start()

        # GPU consumer thread
        consumer = threading.Thread(
            target=self._consumer_iter_worker,
            kwargs=dict(
                sample_queue=sample_queue,
                result_queue=result_queue,
                batch_size=batch_size,
            ),
            daemon=True,
        )
        consumer.start()

        # Yield results as they come
        while True:
            result = result_queue.get()
            if result is None:
                break
            yield result

        producer.join()
        consumer.join()

    def _producer_worker(
        self,
        *,
        pod5_paths: Union[PathLike, PathLikeList],
        sample_queue: "queue.Queue[Optional[Dict]]",
        read_ids: Optional[List[str]],
        progress_bar: Optional[tqdm.tqdm]
    ) -> None:
        """CPU thread: stream reads → standardise → chunk → queue."""
        try:
            with pod5.DatasetReader(
                pod5_paths, recursive=True, max_cached_readers=4, threads=4
            ) as reader:
                reads_iter = (
                    reader.reads(selection={UUID(r) for r in read_ids})
                    if read_ids
                    else reader.reads()
                )

                for rec in reads_iter:
                    try:
                        sample_queue.put(self._process_record(rec), block=True)
                        if progress_bar:
                            progress_bar.update(1)
                    except Exception as exc:
                        logger.warning("Failed on read %s: %s", rec.read_id, exc)
        finally:
            sample_queue.put(None)
            if progress_bar:
                progress_bar.close()

    def _consumer_loop(
        self,
        *,
        sample_queue: "queue.Queue[Optional[Dict]]",
        in_memory_results: Optional[InferenceResults],
        zir_manager: Optional['ZIRShardManager'],
        batch_size: int,
        progress_bar: Optional[tqdm.tqdm]
    ) -> int:
        """Main thread: pop samples, build batches, run GPU, store results."""
        current_batch: List[Dict] = []
        finished = False
        processed_count = 0

        while not finished:
            sample = sample_queue.get()
            if sample is None:
                finished = True
            else:
                current_batch.append(sample)

            if finished or len(current_batch) == batch_size:
                if current_batch:
                    batch_results = self._run_gpu_batch(current_batch)
                    
                    for result in batch_results:
                        if in_memory_results:
                            in_memory_results.add_read_result(result)
                        if zir_manager:
                            zir_manager.add_result(result)
                        if progress_bar:
                            progress_bar.update(1)
                        processed_count += 1
                    
                    current_batch = []
        
        if progress_bar:
            progress_bar.close()
            
        return processed_count

    def _consumer_iter_worker(
        self,
        *,
        sample_queue: "queue.Queue[Optional[Dict]]",
        result_queue: "queue.Queue[Optional[ReadResult]]",
        batch_size: int,
    ) -> None:
        """GPU thread for iterator: process batches and queue results."""
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
                    batch_results = self._run_gpu_batch(current_batch)
                    for result in batch_results:
                        result_queue.put(result)
                    current_batch = []
        
        result_queue.put(None)  # Sentinel

    def _run_gpu_batch(self, batch: List[Dict]) -> List[ReadResult]:
        """Collate, push to GPU, run model, return results."""
        batch_t = collate_fn(batch)

        inputs = {k: v.to(self.device) for k, v in batch_t["inputs"].items()}
        use_amp = (self.config.float_dtype in {"float16", "fp16"}) and (self.device.type != "cpu")
        dtype = torch.float16 if self.config.float_dtype in {"float16", "fp16"} else None
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=use_amp, dtype=dtype
        ):
            outputs = self.model(**inputs)
            if not self.save_raw:
                decoded = crf_smoothing(outputs['segmentation'], batch_t["metadata"]["num_tokens"])

        results = []
        for i, read_id in enumerate(batch_t["metadata"]["read_id"]):
            num_chunks = int(batch_t["metadata"]["num_tokens"][i])
            logits = {
                k: v[i].cpu().numpy()
                for k, v in outputs.items()
            }
            
            result = ReadResult(
                read_id=read_id,
                _logits=logits,
                num_chunks=num_chunks,
                chunk_size=self.config.chunk_size
            )
            if not self.save_raw:
                result = result.to_compressed(k=3, smoothed_preds=decoded[i])
            results.append(result)
        
        return results
    
    def _get_num_reads(
        self,
        read_ids: Optional[List[str]],
        pod5_pathset: PathSet
    ) -> Optional[int]:
        if read_ids:
            return len(read_ids)
        try:
            with pod5.DatasetReader(
                paths=pod5_pathset.paths, recursive=True
            ) as reader:
                return len(reader)
        except Exception as e:
            return None