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
from ..feeders import SequenceStandardizer, collate_fn
from ..storages import InferenceResults, InferenceMetadata, ReadResultDetailed, ReadResult
from ..utils import PathSet
from ..io import ZIRWriter, ZIRShardManager
from ..io.archive_writer import new_run_id, run_id_of_tmp_dir


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

_POLL_S = 0.1
_STOPPED = object()  # returned by _get once the pipeline has been stopped


def _put(q: queue.Queue, item, stop: threading.Event) -> bool:
    """Blocking put that gives up (returns False) once ``stop`` is set."""
    while not stop.is_set():
        try:
            q.put(item, timeout=_POLL_S)
            return True
        except queue.Full:
            continue
    return False


def _get(q: queue.Queue, stop: threading.Event):
    """Blocking get that returns ``_STOPPED`` once ``stop`` is set."""
    while not stop.is_set():
        try:
            return q.get(timeout=_POLL_S)
        except queue.Empty:
            continue
    return _STOPPED


def _raise_worker_error(errors: List[BaseException], processed: Optional[int] = None) -> None:
    if errors:
        done = "" if processed is None else f" after {processed} read(s)"
        raise RuntimeError(f"Inference worker failed{done}: {errors[0]!r}") from errors[0]


class Inference(InferenceBase):
    """
    Stream reads from POD5, preprocess on CPU, batch on-the-fly,
    and run inference on GPU.
    """

    def __init__(
        self,
        config: Union[ModelConfig, str, Dict],
        device: Optional[torch.device] = None,
        save_raw: bool = False,
    ) -> None:
        super().__init__(config=config, device=device)
        self.save_raw = save_raw

    def predict(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        output_path: Optional[Union[str, Path]] = None,
        read_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        shard_size: Optional[int] = None,
        show_progress: bool = True,
        return_results: bool = True,
        staging_dir: Optional[Union[str, Path]] = None,
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
        staging_dir   : Existing .tmp_ dir to write the .zir.partial shards into; the
                        caller then finalizes it (see ZIRShardManager). None = this
                        call creates and finalizes its own
        
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

        # One id per run: an infer-multi worker's chunks share the staging dir's.
        run_id = run_id_of_tmp_dir(staging_dir) if staging_dir else new_run_id()
        metadata = self._build_metadata(pod5_pathset, batch_size, run_id=run_id)
        
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
                staging_dir=staging_dir,
            )

        producer_pbar = None
        consumer_pbar = None
        if show_progress:
            producer_pbar, consumer_pbar = self._make_progress_bars(num_reads)

        # Queue for preprocessed samples
        sample_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(
            maxsize=batch_size * self.queue_size_mult
        )
        
        stats = {"failed": 0}
        stop = threading.Event()
        errors: List[BaseException] = []

        # Producer thread
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                sample_queue=sample_queue,
                progress_bar=producer_pbar,
                stats=stats,
                stop=stop,
                errors=errors,
            ),
            daemon=True,
            name="pod5-producer",
        )
        producer.start()

        processed_count = 0
        succeeded = False
        try:
            processed_count = self._consumer_loop(
                sample_queue=sample_queue,
                in_memory_results=in_memory_results,
                zir_manager=zir_manager,
                batch_size=batch_size,
                progress_bar=consumer_pbar
            )
            succeeded = True
        finally:
            # On failure this unblocks the producer; on success it has already exited.
            stop.set()
            producer.join()
            # Shards only get their final .zir names if the whole run succeeded;
            # otherwise they stay readable as .zir.partial.
            if zir_manager:
                # Final header values; the manager also sets num_reads_processed.
                metadata.total_inference_time = time.time() - start
                if succeeded and not errors:
                    zir_manager.close()
                else:
                    zir_manager.abort()
            for pbar in (producer_pbar, consumer_pbar):
                if pbar:
                    pbar.close()

        if stats["failed"]:
            logger.warning("Failed to process %d read(s)", stats["failed"])

        _raise_worker_error(errors, processed_count)

        # Finalize
        total_time = time.time() - start

        if in_memory_results:
            in_memory_results.metadata.total_inference_time = total_time
            
        logger.info(
            "Done – processed %d reads in %.2fs",
            processed_count,
            total_time,
        )
        
        return in_memory_results

    @staticmethod
    def _make_progress_bars(num_reads: Optional[int]):
        """Return the (reading, processed) progress bars; anything with update()/close() works."""
        bar_fmt = (
            "{desc:<20} "
            "{percentage:6.2f}%|"
            "{bar:30}| "
            "{n_fmt}/{total_fmt} "
            "[{elapsed}/{remaining}, {rate_fmt}]"
        )
        return (
            tqdm(desc="Reading POD5", unit="reads", total=num_reads,
                 leave=True, bar_format=bar_fmt, position=0),
            tqdm(desc="Processed reads", unit="reads", total=num_reads,
                 leave=True, bar_format=bar_fmt, position=1),
        )

    def predict_iter(
        self,
        pod5_paths: Union[PathLike, PathLikeList],
        read_ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Iterator[ReadResultDetailed]:
        """
        Iterator version for streaming results without storing in memory.
        
        Yields
        ------
        ReadResultDetailed objects as they are processed
        
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
        result_queue: "queue.Queue[Optional[ReadResultDetailed]]" = queue.Queue(
            maxsize=batch_size * 2
        )

        stats = {"failed": 0}
        stop = threading.Event()
        errors: List[BaseException] = []

        # Producer thread
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                sample_queue=sample_queue,
                progress_bar=None,
                stats=stats,
                stop=stop,
                errors=errors,
            ),
            daemon=True,
            name="pod5-producer",
        )
        producer.start()

        # GPU consumer thread
        consumer = threading.Thread(
            target=self._consumer_iter_worker,
            kwargs=dict(
                sample_queue=sample_queue,
                result_queue=result_queue,
                batch_size=batch_size,
                stop=stop,
                errors=errors,
            ),
            daemon=True,
            name="gpu-consumer",
        )
        consumer.start()

        # Yield results as they come. The finally block also runs when the caller
        # stops iterating early, so both threads are always shut down.
        try:
            while True:
                result = result_queue.get()
                if result is None:
                    break
                yield result
        finally:
            stop.set()
            producer.join()
            consumer.join()

        if stats["failed"]:
            logger.warning("Failed to process %d read(s)", stats["failed"])

        _raise_worker_error(errors)

    def _producer_worker(
        self,
        *,
        pod5_paths: Union[PathLike, PathLikeList],
        sample_queue: "queue.Queue[Optional[Dict]]",
        read_ids: Optional[List[str]],
        progress_bar: Optional[tqdm.tqdm],
        stats: Optional[Dict[str, int]],
        stop: threading.Event,
        errors: List[BaseException],
    ) -> None:
        """CPU thread: stream reads → standardise → chunk → queue.

        Per-read failures are counted in ``stats``; anything else (e.g. unreadable
        POD5, bad read id) is recorded in ``errors`` for the caller to raise.
        """
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
                    if stop.is_set():
                        return
                    sample = None
                    try:
                        sample = self._process_record(rec)
                    except Exception as exc:
                        stats["failed"] += 1
                        logger.warning("Failed on read %s: %s", rec.read_id, exc)
                    finally:
                        if progress_bar:
                            progress_bar.update(1)
                            
                    if sample is not None and not _put(sample_queue, sample, stop):
                        return

        except Exception as exc:
            errors.append(exc)
        finally:
            _put(sample_queue, None, stop)
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
                            in_memory_results.add_result(result)
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
        result_queue: "queue.Queue[Optional[ReadResultDetailed]]",
        batch_size: int,
        stop: threading.Event,
        errors: List[BaseException],
    ) -> None:
        """GPU thread for iterator: process batches and queue results."""
        current_batch: List[Dict] = []
        try:
            while True:
                sample = _get(sample_queue, stop)
                if sample is _STOPPED:
                    return
                if sample is not None:
                    current_batch.append(sample)

                if current_batch and (sample is None or len(current_batch) == batch_size):
                    for result in self._run_gpu_batch(current_batch):
                        if not _put(result_queue, result, stop):
                            return
                    current_batch = []

                if sample is None:
                    return
        except Exception as exc:
            errors.append(exc)
        finally:
            _put(result_queue, None, stop)  # Sentinel

    def _run_gpu_batch(self, batch: List[Dict]) -> List[ReadResultDetailed]:
        """Collate, push to GPU, run model, return results."""
        batch_t = collate_fn(batch)

        inputs = {k: v.to(self.device) for k, v in batch_t["inputs"].items()}
        use_amp = (self.config.float_dtype in {"float16", "fp16"}) and (self.device.type != "cpu")
        dtype = torch.float16 if self.config.float_dtype in {"float16", "fp16"} else None
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=use_amp, dtype=dtype
        ):
            outputs = self.model(**inputs)

        results = []
        for i, read_id in enumerate(batch_t["metadata"]["read_id"]):
            num_chunks = int(batch_t["metadata"]["num_tokens"][i])
            cropped = bool(batch_t["metadata"]["cropped"][i])
            logits = {
                k: v[i].cpu().numpy()
                for k, v in outputs.items()
            }
            
            result = ReadResultDetailed(
                read_id=read_id,
                _logits=logits,
                num_chunks=num_chunks,
                chunk_size=self.config.chunk_size,
                cropped=cropped
            )
            if not self.save_raw:
                result = result.to_compressed()
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