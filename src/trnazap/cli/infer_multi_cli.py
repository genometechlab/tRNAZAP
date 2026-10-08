"""Multi-GPU infer subcommand for trnazap."""
from __future__ import annotations

import logging
import logging.handlers
import math
import multiprocessing as mp
import sys
from pathlib import Path

import pod5
import torch

from ..inference import Inference
from ..io.archive_writer import finalize_tmp_dir, make_tmp_dir, warn_existing_outputs
from .logging_setup import setup_worker_logging

logger = logging.getLogger(__name__)


def register_subparser(subparsers):
    parser = subparsers.add_parser(
        "infer-multi",
        help="Run inference on tRNA data using multiple GPUs",
        description="Perform inference using trained models across multiple GPUs by sharding read IDs.",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input path: POD5 file or directory containing POD5",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Configuration file (YAML) or model name",
    )

    parser.add_argument(
        "--results-dir",
        "-o",
        type=str,
        default=".",
        help="Directory to write the results_XXXXX.zir files into (default: current dir)",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of reads per chunk (one results_XXXXX.zir each) handed to a GPU worker. "
        "Default: split the reads evenly across the GPUs.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bars",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for each worker (default: 1024)",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: detect from torch.cuda.device_count())",
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save complete model outputs instead of compressed ones",
    )

    parser.set_defaults(func=run_infer_multi)


class _SharedCounter:
    """Progress-bar stand-in for workers: adds to a counter the main process displays."""

    def __init__(self, value):
        self.value = value

    def update(self, n: int = 1) -> None:
        with self.value.get_lock():
            self.value.value += n

    def close(self) -> None:
        pass


class _WorkerInference(Inference):
    """Inference whose progress goes to shared counters instead of its own bars."""

    def __init__(self, *args, counters, **kwargs):
        super().__init__(*args, **kwargs)
        self._counters = counters

    def _make_progress_bars(self, num_reads):
        return tuple(_SharedCounter(v) for v in self._counters)


def _chunk_path(results_dir: Path, shard_idx: int) -> Path:
    return results_dir / f"results_{shard_idx:05d}.zir"


def _gpu_worker(
    gpu_idx: int,
    job_queue,
    log_queue,
    counters,
    input_path: Path,
    config_path: str,
    results_dir: Path,
    tmp_dir: Path,
    batch_size: int,
    save_raw: bool,
):
    setup_worker_logging(log_queue, prefix=f"[GPU {gpu_idx}] ")
    shard_idx = None
    try:
        torch.cuda.set_device(gpu_idx)
        engine = _WorkerInference(
            config_path, device=f"cuda:{gpu_idx}", save_raw=save_raw, counters=counters
        )
        while (job := job_queue.get()) is not None:
            shard_idx, read_ids = job
            logger.info("Processing chunk %d (%d reads)", shard_idx, len(read_ids))
            engine.predict(
                pod5_paths=input_path,
                read_ids=read_ids,
                batch_size=batch_size,
                show_progress=True,  # feeds the shared counters
                return_results=False,
                output_path=_chunk_path(results_dir, shard_idx),
                shard_size=None,
                staging_dir=tmp_dir,
            )
            shard_idx = None
    except KeyboardInterrupt:
        sys.exit(130)
    except BaseException:
        where = "" if shard_idx is None else f" on chunk {shard_idx}"
        logger.exception("Worker failed%s", where)
        sys.exit(1)


def run_infer_multi(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    visible_gpus = torch.cuda.device_count()
    num_gpus = args.gpus or visible_gpus
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for multi-GPU inference.")
    if num_gpus > visible_gpus:
        raise RuntimeError(f"Asked for {num_gpus} GPUs but only {visible_gpus} are visible.")

    # ------------------------------------------------------------------
    # 1) collect read IDs once (main process) and split them into chunks
    # ------------------------------------------------------------------
    logger.info("Collecting read IDs …")
    read_ids_set = set()
    with pod5.DatasetReader(input_path, recursive=True) as rdr:
        for rec in rdr:
            read_ids_set.add(str(rec.read_id))
    read_ids = list(read_ids_set)
    total_reads = len(read_ids)
    if total_reads == 0:
        raise ValueError(f"No reads found in {input_path}")

    step = args.shard_size or math.ceil(total_reads / num_gpus)
    shards = [read_ids[i : i + step] for i in range(0, total_reads, step)]
    num_workers = min(num_gpus, len(shards))

    logger.info("Total reads: %d", total_reads)
    logger.info("Chunks     : %d", len(shards))
    logger.info("GPUs       : %d", num_workers)

    # ------------------------------------------------------------------
    # 2) one temp dir for the whole run; workers write .zir.partial into it
    # ------------------------------------------------------------------
    warn_existing_outputs(results_dir, sorted(results_dir.glob("results_[0-9]*.zir")))
    tmp_dir = make_tmp_dir(results_dir)

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for idx, shard in enumerate(shards):
        job_queue.put((idx, shard))
    for _ in range(num_workers):
        job_queue.put(None)  # one stop marker per worker

    log_queue = ctx.Queue()
    listener = logging.handlers.QueueListener(
        log_queue, *logging.getLogger().handlers, respect_handler_level=True
    )
    counters = (ctx.Value("q", 0), ctx.Value("q", 0))  # reads read, reads processed

    bars = []
    if not args.no_progress:
        bars = list(Inference._make_progress_bars(total_reads))

    def sync_bars():
        for bar, value in zip(bars, counters):
            bar.update(value.value - bar.n)

    # ------------------------------------------------------------------
    # 3) run the workers
    # ------------------------------------------------------------------
    listener.start()
    processes = []
    try:
        for gpu_idx in range(num_workers):
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gpu_idx,
                    job_queue,
                    log_queue,
                    counters,
                    input_path,
                    args.config,
                    results_dir,
                    tmp_dir,
                    args.batch_size,
                    args.save_raw,
                ),
                name=f"gpu{gpu_idx}",
            )
            p.start()
            processes.append(p)

        for p in processes:
            while p.is_alive():
                p.join(timeout=0.5)
                sync_bars()
        sync_bars()
    except BaseException:
        # e.g. Ctrl-C: the workers get it too; give them a moment, then stop them.
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        job_queue.cancel_join_thread()
        raise
    finally:
        for bar in bars:
            bar.close()
        listener.stop()

    # ------------------------------------------------------------------
    # 4) finalize only if every worker succeeded
    # ------------------------------------------------------------------
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        job_queue.cancel_join_thread()
        logger.error(
            "GPU worker(s) %s failed (exit codes %s); nothing was moved to %s. "
            "Results written so far are left as .zir.partial in %s",
            failed, [processes[i].exitcode for i in failed], results_dir, tmp_dir,
        )
        return 1

    finals = finalize_tmp_dir(tmp_dir, results_dir)
    logger.info("All %d chunks processed → %s", len(finals), results_dir)
    return 0
