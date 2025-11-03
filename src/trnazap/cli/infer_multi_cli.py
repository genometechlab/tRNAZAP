"""Multi-GPU infer subcommand for trnazap."""
from __future__ import annotations

import logging
from pathlib import Path
import pickle
import multiprocessing as mp
import queue
import torch
import pod5

from ..inference import Inference  # same import style as your single-GPU CLI

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
        help="Directory to write shard ZIR files into (default: current dir)",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000000,
        help="Number of reads per shard to assign to a GPU worker. If omitted, all reads go in one shard.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars inside each worker",
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


def _run_chunk(
    shard_idx: int,
    read_ids: list[str],
    gpu_idx: int,
    input_path: Path,
    config_path: str,
    results_dir: Path,
    batch_size: int,
    save_raw: bool,
    show_progress: bool,
):
    # pin to gpu
    torch.cuda.set_device(gpu_idx)
    device = f"cuda:{gpu_idx}"

    if gpu_idx==0:
        logger.info("[GPU %d] Processing chunk %d (%d reads)", gpu_idx, shard_idx, len(read_ids))

    out_file = results_dir / f"results_{shard_idx}.zir"

    engine = Inference(config_path, device=device, save_raw=save_raw)
    engine.predict(
        pod5_paths=input_path,
        read_ids=read_ids,
        batch_size=batch_size,
        show_progress=gpu_idx==0,
        return_results=False,
        output_path=out_file,
        shard_size=None,
    )

    if gpu_idx==0:
        logger.info("[GPU %d] Finished chunk %d → %s", gpu_idx, shard_idx, out_file)


def _gpu_worker(
    gpu_idx: int,
    job_queue: mp.Queue,
    input_path: Path,
    config_path: str,
    results_dir: Path,
    batch_size: int,
    save_raw: bool,
    no_progress: bool,
):
    # each process should at least have basic logging configured
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    )

    while True:
        try:
            shard_idx, shard_reads = job_queue.get(timeout=3)
        except queue.Empty:
            break

        _run_chunk(
            shard_idx=shard_idx,
            read_ids=shard_reads,
            gpu_idx=gpu_idx,
            input_path=input_path,
            config_path=config_path,
            results_dir=results_dir,
            batch_size=batch_size,
            save_raw=save_raw,
            show_progress=(not no_progress) and (gpu_idx == 0),
        )


def run_infer_multi(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


    visible_gpus = torch.cuda.device_count()
    num_gpus = args.gpus or visible_gpus
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for multi-GPU inference.")
    logger.info("GPUs       : %d", num_gpus)
    
    # ------------------------------------------------------------------
    # 1) collect / load read IDs once (main process)
    # ------------------------------------------------------------------
    logger.info("Collecting read IDs …")
    read_ids_set = set()
    with pod5.DatasetReader(input_path, recursive=True) as rdr:
        for rec in rdr:
            read_ids_set.add(str(rec.read_id))
    read_ids = list(read_ids_set)

    total_reads = len(read_ids)
    step = args.shard_size or total_reads//visible_gpus
    shards = [read_ids[i : i + step] for i in range(0, total_reads, step)]

    logger.info("Total reads: %d", total_reads)
    logger.info("Chunks     : %d", len(shards))

    # ------------------------------------------------------------------
    # 2) create job queue
    # ------------------------------------------------------------------
    mp.set_start_method("spawn", force=True)
    job_queue = mp.Queue()
    for idx, shard in enumerate(shards):
        job_queue.put((idx, shard))

    # ------------------------------------------------------------------
    # 3) start workers
    # ------------------------------------------------------------------
    processes = []
    for gpu_idx in range(num_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gpu_idx,
                job_queue,
                input_path,
                args.config,
                results_dir,
                args.batch_size,
                args.save_raw,
                args.no_progress,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("All chunks processed.")
    return 0
