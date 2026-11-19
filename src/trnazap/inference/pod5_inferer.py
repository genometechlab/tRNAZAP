from __future__ import annotations

import bisect
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
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

# Token-count boundaries of the length buckets: reads are batched only with reads
# of a similar length, so a batch is not padded to an unrelated long read. Chosen
# from the read-length distributions of E. coli, yeast and human data at strides
# 54 and 36 (notebooks/length_distribution.ipynb): steps of 16 where most reads
# are, coarser steps in the tail. 20 boundaries = 21 buckets.
DEFAULT_LENGTH_BUCKETS: Tuple[int, ...] = (
    96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384, 448, 512, 640, 768, 1024,
)
# 0 = every bucket uses batch_size (see bucket_batch_sizes)
DEFAULT_BUCKET_BATCH_SCALING = 0.0


def normalize_length_buckets(length_buckets: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Validate bucket boundaries; ``None`` or empty means a single bucket."""
    if not length_buckets:
        return ()
    bounds = tuple(int(b) for b in length_buckets)
    if any(b <= 0 for b in bounds):
        raise ValueError(f"length_buckets must be positive token counts, got {list(bounds)}")
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"length_buckets must be strictly increasing, got {list(bounds)}")
    return bounds


def bucket_batch_sizes(
    boundaries: Sequence[int],
    batch_size: int,
    max_seq_len: int,
    scaling: float = DEFAULT_BUCKET_BATCH_SCALING,
) -> List[int]:
    """Batch size of each bucket (one per bucket, ``len(boundaries) + 1``).

    With ``scaling`` m, bucket i (upper boundary U_i, the last one ``max_seq_len``)
    gets ``batch_size * (1 + m * (max_seq_len / U_i - 1))``, rounded. The longest
    bucket always gets ``batch_size``. m=0: every bucket uses ``batch_size``;
    m=1: every full batch holds about ``batch_size * max_seq_len`` tokens, the
    same as a full batch of the longest reads.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if scaling < 0:
        raise ValueError(f"bucket_batch_scaling must be >= 0, got {scaling}")
    uppers = [min(b, max_seq_len) for b in boundaries] + [max_seq_len]
    return [
        max(1, round(batch_size * (1 + scaling * (max_seq_len / u - 1))))
        for u in uppers
    ]


def parse_length_buckets(text: str) -> Tuple[int, ...]:
    """Parse a CLI value: comma-separated token counts, or "none" for one bucket."""
    import argparse

    if text.strip().lower() in {"none", ""}:
        return ()
    try:
        return normalize_length_buckets([int(x) for x in text.split(",") if x.strip()])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated increasing token counts or 'none', got {text!r} ({exc})"
        ) from None


def add_bucketing_arguments(parser) -> None:
    """Add --length-buckets and --bucket-batch-scaling to an argparse parser."""
    default = ", ".join(map(str, DEFAULT_LENGTH_BUCKETS))
    parser.add_argument(
        "--length-buckets",
        type=parse_length_buckets,
        default=DEFAULT_LENGTH_BUCKETS,
        metavar="T1,T2,...|none",
        help="Token-count boundaries for grouping reads of similar length into the same "
        "batch, which cuts padding. Predictions are unchanged (logits can differ by "
        "float rounding); output order no longer follows the POD5 order. "
        "'none' = no grouping. "
        f"Default: {default}",
    )
    parser.add_argument(
        "--bucket-batch-scaling",
        type=float,
        default=DEFAULT_BUCKET_BATCH_SCALING,
        metavar="M",
        help="[EXPERIMENTAL] Use larger batches for shorter reads. 0 = off (default): "
        "every batch uses --batch-size. 1 = every full batch holds about as many "
        "tokens as a full batch of the longest reads. Values in between are more "
        "conservative; values above 1 exceed that size (warned). The longest reads "
        "always use --batch-size. More reads are held in memory while batches "
        "fill (see README).",
    )


class _LengthBucketer:
    """Collects samples into length buckets and hands back each bucket's batch once full."""

    def __init__(self, boundaries: Sequence[int], batch_sizes: Sequence[int]) -> None:
        assert len(batch_sizes) == len(boundaries) + 1
        self.boundaries = list(boundaries)
        self.batch_sizes = list(batch_sizes)
        self.buckets: List[List[Dict]] = [[] for _ in batch_sizes]

    def add(self, sample: Dict) -> Optional[List[Dict]]:
        """Add a sample; return its bucket's batch if that filled it, else None."""
        i = bisect.bisect_right(self.boundaries, sample["metadata"]["num_tokens"])
        bucket = self.buckets[i]
        bucket.append(sample)
        if len(bucket) < self.batch_sizes[i]:
            return None
        self.buckets[i] = []
        return bucket

    def flush(self) -> Iterator[List[Dict]]:
        """Yield the leftover (partial) batches, shortest bucket first."""
        for i, bucket in enumerate(self.buckets):
            if bucket:
                self.buckets[i] = []
                yield bucket


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
        length_buckets: Optional[Sequence[int]] = DEFAULT_LENGTH_BUCKETS,
        bucket_batch_scaling: float = DEFAULT_BUCKET_BATCH_SCALING,
    ) -> None:
        """
        Args
        ----
        config        : ModelConfig, path to a YAML config, model name, or dict
        device        : Torch device; None = CUDA if available, else CPU
        save_raw      : Keep full ReadResultDetailed reads (default: compress to ReadResult)
        length_buckets: Token-count boundaries; reads are only batched with reads
                        of the same bucket, which cuts padding. None = one bucket
                        (batch in POD5 order). Results come out grouped by
                        bucket, not in POD5 order
        bucket_batch_scaling: [EXPERIMENTAL] Grow the batch size of shorter
                        buckets relative to predict()'s batch_size (see
                        bucket_batch_sizes). 0 = off; 1 = every full batch holds
                        about as many tokens as a full batch of the longest
                        reads. The longest bucket always uses batch_size.
                        Raises how many reads wait in memory
        """
        # Validate before loading the model, so bad values fail fast
        self.length_buckets: Tuple[int, ...] = normalize_length_buckets(length_buckets)
        if bucket_batch_scaling < 0:
            raise ValueError(f"bucket_batch_scaling must be >= 0, got {bucket_batch_scaling}")
        if bucket_batch_scaling and not self.length_buckets:
            logger.warning("bucket_batch_scaling has no effect without length_buckets")
        if bucket_batch_scaling > 1:
            logger.warning(
                "bucket_batch_scaling=%s > 1: batches of short reads will hold more tokens "
                "than a full batch of the longest reads",
                bucket_batch_scaling,
            )
        self.bucket_batch_scaling = float(bucket_batch_scaling)

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
        batch_size    : Number of reads per GPU batch (per bucket; shorter buckets
                        get more with bucket_batch_scaling, see __init__)
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
        bucketing = self._bucketing(batch_size)
                     
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

        # Queue of collated batches, ready for the model
        batch_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=self.queue_size_mult)

        stats = {"failed": 0}
        stop = threading.Event()
        errors: List[BaseException] = []

        # Producer thread
        producer = threading.Thread(
            target=self._producer_worker,
            kwargs=dict(
                pod5_paths=pod5_pathset.paths,
                read_ids=read_ids,
                batch_queue=batch_queue,
                bucketing=bucketing,
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
                batch_queue=batch_queue,
                in_memory_results=in_memory_results,
                zir_manager=zir_manager,
                progress_bar=consumer_pbar,
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

    def _bucketing(self, batch_size: int) -> Tuple[Tuple[int, ...], List[int]]:
        """(boundaries, per-bucket batch sizes) for a run with this batch_size."""
        bounds, bucket_batch_scaling = self.length_buckets, self.bucket_batch_scaling
        sizes = bucket_batch_sizes(bounds, batch_size, self.config.max_seq_len, bucket_batch_scaling)
        if bucket_batch_scaling and bounds:
            edges = (0,) + bounds
            logger.info(
                "Bucket batch scaling %s (experimental), batch size per bucket: %s",
                bucket_batch_scaling,
                ", ".join(
                    f"[{lo},{hi})→{b}"
                    for lo, hi, b in zip(edges, bounds + ("inf",), sizes)
                ),
            )
        return bounds, sizes

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
        ReadResultDetailed objects as they are processed (grouped by length
        bucket, not in POD5 order; see ``__init__`` for the bucketing settings)
        
        Example
        -------
        for result in inference.predict_iter("data.pod5"):
            if result.get_preds('classification') == 0:
                print(f"Found target read: {result.read_id}")
        """
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        bucketing = self._bucketing(batch_size)

        # Queue of collated batches, ready for the model
        batch_queue: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=self.queue_size_mult)

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
                batch_queue=batch_queue,
                bucketing=bucketing,
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
                batch_queue=batch_queue,
                result_queue=result_queue,
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
        batch_queue: "queue.Queue[Optional[Dict]]",
        bucketing: Tuple[Sequence[int], Sequence[int]],
        read_ids: Optional[List[str]],
        progress_bar: Optional[tqdm.tqdm],
        stats: Optional[Dict[str, int]],
        stop: threading.Event,
        errors: List[BaseException],
    ) -> None:
        """CPU thread: stream reads → standardise → length bucket → collate → queue.

        Each read goes into the bucket of its token count; a bucket that reaches
        its batch size is collated and queued as one batch. At the end of the
        input the leftover partial buckets are queued too. Reads therefore reach
        the model grouped by length, not in POD5 order.

        Memory: besides the queued batches, up to sum(batch_size_i - 1) reads
        can wait in partly filled buckets (num_buckets * (batch_size - 1)
        without bucket batch scaling).

        Per-read failures are counted in ``stats``; anything else (e.g. unreadable
        POD5, bad read id) is recorded in ``errors`` for the caller to raise.
        """
        bucketer = _LengthBucketer(*bucketing)
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
                            
                    if sample is None:
                        continue
                    batch = bucketer.add(sample)
                    if batch is not None and not _put(batch_queue, collate_fn(batch), stop):
                        return

                for batch in bucketer.flush():
                    if not _put(batch_queue, collate_fn(batch), stop):
                        return

        except Exception as exc:
            errors.append(exc)
        finally:
            _put(batch_queue, None, stop)
            if progress_bar:
                progress_bar.close()

    def _consumer_loop(
        self,
        *,
        batch_queue: "queue.Queue[Optional[Dict]]",
        in_memory_results: Optional[InferenceResults],
        zir_manager: Optional['ZIRShardManager'],
        progress_bar: Optional[tqdm.tqdm],
    ) -> int:
        """Main thread: pop collated batches, run the model, store results."""
        processed_count = 0

        while (batch := batch_queue.get()) is not None:
            for result in self._run_gpu_batch(batch):
                if in_memory_results:
                    in_memory_results.add_result(result)
                if zir_manager:
                    zir_manager.add_result(result)
                if progress_bar:
                    progress_bar.update(1)
                processed_count += 1

        if progress_bar:
            progress_bar.close()
            
        return processed_count

    def _consumer_iter_worker(
        self,
        *,
        batch_queue: "queue.Queue[Optional[Dict]]",
        result_queue: "queue.Queue[Optional[ReadResultDetailed]]",
        stop: threading.Event,
        errors: List[BaseException],
    ) -> None:
        """GPU thread for iterator: run collated batches and queue results."""
        try:
            while True:
                batch = _get(batch_queue, stop)
                if batch is None or batch is _STOPPED:
                    return
                for result in self._run_gpu_batch(batch):
                    if not _put(result_queue, result, stop):
                        return
        except Exception as exc:
            errors.append(exc)
        finally:
            _put(result_queue, None, stop)  # Sentinel

    def _run_gpu_batch(self, batch_t: Dict) -> List[ReadResultDetailed]:
        """Push a collated batch (see collate_fn) to the device, run the model, return results."""
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