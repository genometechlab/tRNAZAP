"""Infer subcommand for trnazap."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from ..inference import Inference

logger = logging.getLogger(__name__)

def register_subparser(subparsers):
    """Register the infer subcommand."""
    parser = subparsers.add_parser(
        "infer",
        help="Run inference on tRNA data",
        description="Perform inference using trained models",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input path: POD5 file, multiple POD5 files, or a directory containing POD5",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Configuration file (YAML) or model name",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output ZIR file or directory for sharded output; if omitted, results stay in memory",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU/CPU inference (default: 32)",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of reads per output shard when writing ZIR; default: single file",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save complete model outputs",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only inference",
    )

    parser.set_defaults(func=run_infer)


def run_infer(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    config_path = args.config

    # decide device
    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = None

    # create inference object
    inference = Inference(
        config=config_path,
        device=device,
        save_raw=args.save_raw,
    )

    # run prediction
    inference.predict(
        pod5_paths=input_path,
        output_path=args.output,
        read_ids=None,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        show_progress=not args.no_progress,
        return_results=False,
    )
    
    return 0