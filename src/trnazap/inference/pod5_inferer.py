from __future__ import annotations

import bisect
import logging
import multiprocessing
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
from uuid import UUID
from dataclasses import dataclass, field

import numpy as np
import pod5
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from .inference_base import InferenceBase, preprocess_signal
from ..config.model_config import ModelConfig, ModelLoader
from ..feeders import SequenceStandardizer, collate_fn
from ..storages import InferenceResults, InferenceMetadata, ReadResultDetailed, ReadResult
from ..utils import PathSet
from ..io import ZIRWriter, ZIRShardManager
from ..io.archive_writer import new_run_id, run_id_of_tmp_dir


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
PathLikeList = Union[PathLike, List[PathLike]]

# Processes that read and prepare batches while the GPU runs (0 = do it in the
# main process, one batch after another; slower, for debugging)
DEFAULT_NUM_WORKERS = 4

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


def add_batching_arguments(parser) -> None:
    """Add --num-workers, --length-buckets and --bucket-batch-scaling to an argparse parser."""
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        metavar="N",
        help="Processes that read and prepare reads while the model runs "
        f"(default: {DEFAULT_NUM_WORKERS}). 0 = do it in the main process, one batch "
        "after another (slower; for debugging).",
    )
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


def collect_read_ids(pod5_paths) -> List[str]:
    """All read ids in ``pod5_paths`` (files and/or directories, searched recursively).

    Files are taken in sorted order and reads in on-disk order, duplicates
    dropped, so the same input always gives the same list. Inference and
    infer-multi split the work by this list, never by position in the POD5
    stream: pod5 can return the files of a multi-file dataset in a different
    order each time it is opened.
    """
    with pod5.DatasetReader(pod5_paths, recursive=True, threads=1) as reader:
        return list(dict.fromkeys(str(r) for r in reader.read_ids))


def _add(counter, n: int) -> None:
    """Add to a shared multiprocessing counter (see _Pod5BatchDataset)."""
    if n:
        with counter.get_lock():
            counter.value += n


class _Pod5BatchDataset(IterableDataset):
    """Reads POD5 and yields collated, length-grouped batches ready for the model.

    Runs inside DataLoader worker processes, so it holds only what preprocessing
    needs (paths, read ids, settings), never the model. With K workers, worker k
    takes every K-th read id of ``read_ids`` and asks pod5 for exactly those
    reads, so the workers cover all reads exactly once. (Splitting by position
    in the POD5 stream instead is unsafe: see collect_read_ids.)

    Each worker groups its own reads: a length bucket that reaches its batch size
    is collated and yielded as one batch, and at the end of the worker's reads
    its leftover partial buckets are yielded too. Reads therefore reach the model
    grouped by length, not in POD5 order.

    Memory: per worker, up to sum(batch_size_i - 1) reads can wait in partly
    filled buckets (num_buckets * (batch_size - 1) without bucket batch scaling),
    plus the batches the DataLoader prefetches.

    ``counters`` = (reads read, reads failed): shared values, updated from the
    workers, for the progress bar and the failed-read warning.
    """

    _COUNT_EVERY = 256  # reads between progress-counter updates

    def __init__(
        self,
        pod5_paths: List[str],
        read_ids: List[str],
        *,
        preprocess_params: Dict,
        boundaries: Sequence[int],
        batch_sizes: Sequence[int],
        counters,
    ) -> None:
        super().__init__()
        self.pod5_paths = pod5_paths
        self.read_ids = list(read_ids)  # every read to process (see collect_read_ids)
        self.preprocess_params = preprocess_params
        self.boundaries = list(boundaries)
        self.batch_sizes = list(batch_sizes)
        self.counters = counters

    def __iter__(self) -> Iterator[Dict]:
        info = get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info else (0, 1)
        reads_seen, reads_failed = self.counters
        bucketer = _LengthBucketer(self.boundaries, self.batch_sizes)
        uncounted = 0

        with pod5.DatasetReader(
            self.pod5_paths, recursive=True, max_cached_readers=4, threads=2
        ) as reader:
            mine = self.read_ids[worker::num_workers]
            records = iter(()) if not mine else reader.reads(
                selection={UUID(r) for r in mine}
            )

            for rec in records:
                sample = None
                try:
                    sample = preprocess_signal(
                        rec.signal, str(rec.read_id), **self.preprocess_params
                    )
                except Exception as exc:
                    _add(reads_failed, 1)
                    logger.warning("Failed on read %s: %s", rec.read_id, exc)
                uncounted += 1
                if uncounted >= self._COUNT_EVERY:
                    _add(reads_seen, uncounted)
                    uncounted = 0

                if sample is None:
                    continue
                batch = bucketer.add(sample)
                if batch is not None:
                    yield collate_fn(batch)

        _add(reads_seen, uncounted)
        for batch in bucketer.flush():
            yield collate_fn(batch)


def _loader_batches(loader: DataLoader) -> Iterator[Dict]:
    """Iterate a DataLoader; errors from reading/preprocessing are re-raised as
    ``RuntimeError("Inference worker failed: ...")``. Use with ``contextlib.closing``
    so the worker processes are shut down even when the caller stops early."""
    it = iter(loader)
    try:
        while True:
            try:
                batch = next(it)
            except StopIteration:
                return
            except Exception as exc:
                raise RuntimeError(f"Inference worker failed: {exc!r}") from exc
            yield batch
    finally:
        shutdown = getattr(it, "_shutdown_workers", None)  # multi-process iterators only
        if shutdown is not None:
            shutdown()


class Inference(InferenceBase):
    """
    Stream reads from POD5 and run the model on them.

    DataLoader worker processes read the POD5 files, preprocess each read,
    group reads of similar length and collate them into batches
    (_Pod5BatchDataset); the main process only runs the model on those
    batches and turns the outputs into per-read results (_run_gpu_batch).
    """

    # Log the batching settings (warnings, bucket batch sizes)? infer-multi turns
    # this off on all GPUs but the first, which run with the same settings
    _log_settings = True

    def __init__(
        self,
        config: Union[ModelConfig, str, Dict],
        device: Optional[torch.device] = None,
        save_raw: bool = False,
        length_buckets: Optional[Sequence[int]] = DEFAULT_LENGTH_BUCKETS,
        bucket_batch_scaling: float = DEFAULT_BUCKET_BATCH_SCALING,
        num_workers: int = DEFAULT_NUM_WORKERS,
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
        num_workers   : Processes that read and prepare batches while the model
                        runs (torch DataLoader workers). 0 = prepare them in the
                        main process, one batch after another (slower; for
                        debugging)
        """
        # Validate before loading the model, so bad values fail fast
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        self.num_workers = int(num_workers)
        self.length_buckets: Tuple[int, ...] = normalize_length_buckets(length_buckets)
        if bucket_batch_scaling < 0:
            raise ValueError(f"bucket_batch_scaling must be >= 0, got {bucket_batch_scaling}")
        if bucket_batch_scaling and not self.length_buckets and self._log_settings:
            logger.warning("bucket_batch_scaling has no effect without length_buckets")
        if bucket_batch_scaling > 1 and self._log_settings:
            logger.warning(
                "bucket_batch_scaling=%s > 1: batches of short reads will hold more tokens "
                "than a full batch of the longest reads",
                bucket_batch_scaling,
            )
        self.bucket_batch_scaling = float(bucket_batch_scaling)
        self._logged_batch_sizes: set = set()  # batch sizes whose bucket table was logged

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

        start = time.time()
        pod5_pathset: PathSet = self._resolve_paths(pod5_paths)
        read_ids = self._resolve_read_ids(read_ids, pod5_pathset)
        num_reads = len(read_ids)

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

        counters = self._make_counters()
        loader = self._make_loader(pod5_pathset, read_ids, batch_size, counters)

        processed_count = 0
        reads_seen = 0  # last value shown on the reading bar
        succeeded = False
        try:
            with closing(_loader_batches(loader)) as batches:
                for batch in batches:
                    for result in self._run_gpu_batch(batch):
                        if in_memory_results:
                            in_memory_results.add_result(result)
                        if zir_manager:
                            zir_manager.add_result(result)
                        if consumer_pbar:
                            consumer_pbar.update(1)
                        processed_count += 1
                    if producer_pbar:
                        reads_seen = self._sync_bar(producer_pbar, counters[0], reads_seen)
            if producer_pbar:
                self._sync_bar(producer_pbar, counters[0], reads_seen)
            succeeded = True
        finally:
            # Shards only get their final .zir names if the whole run succeeded;
            # otherwise they stay readable as .zir.partial.
            if zir_manager:
                # Final header values; the manager also sets num_reads_processed.
                metadata.total_inference_time = time.time() - start
                if succeeded:
                    zir_manager.close()
                else:
                    zir_manager.abort()
            for pbar in (producer_pbar, consumer_pbar):
                if pbar:
                    pbar.close()

        self._warn_failed(counters)

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
    def _make_counters():
        """Shared (reads read, reads failed) counters the loader workers update."""
        return multiprocessing.Value("q", 0), multiprocessing.Value("q", 0)

    @staticmethod
    def _sync_bar(bar, counter, shown: int) -> int:
        """Advance ``bar`` to ``counter``'s value; return the value now shown."""
        value = counter.value
        if value > shown:
            bar.update(value - shown)
        return max(value, shown)

    @staticmethod
    def _warn_failed(counters) -> None:
        failed = counters[1].value
        if failed:
            logger.warning("Failed to process %d read(s)", failed)

    def _make_loader(
        self,
        pod5_pathset: PathSet,
        read_ids: List[str],
        batch_size: int,
        counters,
    ) -> DataLoader:
        """DataLoader yielding collated, length-grouped batches (see _Pod5BatchDataset)."""
        boundaries, batch_sizes = self._bucketing(batch_size)
        dataset = _Pod5BatchDataset(
            [str(p) for p in pod5_pathset.paths],
            read_ids,
            preprocess_params=self._preprocess_params,
            boundaries=boundaries,
            batch_sizes=batch_sizes,
            counters=counters,
        )
        return DataLoader(
            dataset,
            batch_size=None,  # the dataset already yields whole batches
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _bucketing(self, batch_size: int) -> Tuple[Tuple[int, ...], List[int]]:
        """(boundaries, per-bucket batch sizes) for a run with this batch_size."""
        bounds, bucket_batch_scaling = self.length_buckets, self.bucket_batch_scaling
        sizes = bucket_batch_sizes(bounds, batch_size, self.config.max_seq_len, bucket_batch_scaling)
        if bucket_batch_scaling and bounds and self._log_settings and batch_size not in self._logged_batch_sizes:
            self._logged_batch_sizes.add(batch_size)
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
        read_ids = self._resolve_read_ids(read_ids, pod5_pathset)
        counters = self._make_counters()
        loader = self._make_loader(pod5_pathset, read_ids, batch_size, counters)

        # closing() also runs when the caller stops iterating early, so the
        # loader's worker processes are always shut down.
        with closing(_loader_batches(loader)) as batches:
            for batch in batches:
                yield from self._run_gpu_batch(batch)

        self._warn_failed(counters)

    def _run_gpu_batch(self, batch_t: Dict) -> List[ReadResultDetailed]:
        """Push a collated batch (see collate_fn) to the device, run the model, return results."""
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch_t["inputs"].items()}
        use_amp = (self.config.float_dtype in {"float16", "fp16"}) and (self.device.type != "cpu")
        dtype = torch.float16 if self.config.float_dtype in {"float16", "fp16"} else None
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type, enabled=use_amp, dtype=dtype
        ):
            outputs = self.model(**inputs)

        # One device->host copy per output for the whole batch (a copy per read
        # makes the CPU wait on the GPU once per read and output).
        host = {k: v.cpu().numpy() for k, v in outputs.items()}
        meta = batch_t["metadata"]
        num_tokens = meta["num_tokens"].tolist()

        results = []
        for i, read_id in enumerate(meta["read_id"]):
            num_chunks = num_tokens[i]
            cropped = bool(meta["cropped"][i])
            # Each read gets its own copy, so it doesn't keep the whole batch
            # array alive. Per-token outputs ([B, T, C]) are cut to the read's
            # own tokens, dropping the batch padding.
            logits = {
                k: (a[i, :num_chunks] if a.ndim == 3 else a[i]).copy()
                for k, a in host.items()
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
    
    @staticmethod
    def _resolve_read_ids(
        read_ids: Optional[List[str]],
        pod5_pathset: PathSet,
    ) -> List[str]:
        """The given read ids, or all read ids in the input if none (or an empty list) are given."""
        if read_ids:
            return list(read_ids)
        return collect_read_ids([str(p) for p in pod5_pathset.paths])