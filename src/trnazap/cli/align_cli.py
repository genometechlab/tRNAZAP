# src/trnazap/cli/align_cli.py

import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

import shutil
import numba
numba.set_num_threads(1)

from ..aligner.zap_aligner import run_align

"""Parent Module for executing inference and alignment of tRNA.

This module contains argument parsing, overall function control,
and executes submodules.
"""


def register_subparser(subparsers):
    """Register the align subcommand."""
    parser = subparsers.add_parser(
        "align",
        help="Align tRNA sequences",
        description="Align tRNA sequences to reference databases",
    )

    parser.add_argument(
        "--unaligned_bam",
        "-ub",
        type=str,
        required=True,
        help="Basecalled bam paired with pod5 input for model inference",
    )

    parser.add_argument(
        "--inference",
        "-i",
        type=str,
        required=True,
        nargs='*',
        help="tRNA model inference results, multiple inference files can be provided"
        + " such as from the results of mulitple sequencing runs",
    )

    parser.add_argument(
        "--out_dir",
        "-od",
        type=str,
        required=True,
        help="Output directory, if the directory does not exist an attempt"
        + " will be made to create the directory",
    )

    parser.add_argument(
        "--out_pre",
        "-op",
        type=str,
        required=True,
        help="Prefix to be appended before output files",
    )

    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        required=False,
        default=8,  # Get number of available threads
    )

    parser.add_argument(
        "--secondary",
        "-s",
        default=False,
        action="store_true",
        help="Perform alignemnts for second highest classification" +
        " select the better alignment of the two.",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=False,
        default="e_coli",
        choices=["yeast", "e_coli"],
        help="Target substrate. Currently two models are supported:"
        + " yeast tRNA (yeast) and E. coli tRNA (e_coli).",
    )

    parser.add_argument(
        "--ivt_alignment",
        default=False,
        action="store_true",
        help="Use the in vitro transcribed (IVT) scoring profile for the"
        + " selected substrate instead of the biological one. IVT tRNA is"
        + " unmodified and basecalls more cleanly than biological tRNA.",
    )

    parser.add_argument(
        "--pickled_inf_obj",
        default=False,
        action="store_true",
        help="A pre pickled inference obj to reduce repeat run speed"
    )

    # Scoring parameters. These all default to None, meaning 'take the value
    # from the substrate's profile in alignment_defaults.py'. Any value passed
    # here overrides that one field of the profile.
    parser.add_argument(
        "--ident_threshold",
        type=float,
        default=None,
        required=False,
        help="Minimum identity threshold for reads to be considered"
        + " (default: per-model profile)"
    )

    # Wagner-Fisher alignment parameters, edit costs where positive penalizes
    parser.add_argument("--wf_gap_open", type=float, default=None,
                       help="Wagner-Fisher gap open cost, positive penalizes"
                       + " (default: per-model profile)")
    parser.add_argument("--wf_gap_extend", type=float, default=None,
                       help="Wagner-Fisher gap extend cost, positive penalizes"
                       + " (default: per-model profile)")

    # Smith-Waterman alignment parameters, scores where negative penalizes
    parser.add_argument("--sw_gap_open", type=float, default=None,
                       help="Smith-Waterman gap open score, negative penalizes"
                       + " (default: per-model profile)")
    parser.add_argument("--sw_gap_extend", type=float, default=None,
                       help="Smith-Waterman gap extend score, negative penalizes"
                       + " (default: per-model profile)")
    parser.add_argument("--sw_match", type=float, default=None,
                       help="Smith-Waterman match score, positive rewards"
                       + " (default: per-model profile)")
    parser.add_argument("--sw_mismatch", type=float, default=None,
                       help="Smith-Waterman mismatch score, negative penalizes"
                       + " (default: per-model profile)")

    # Set the function to call when this subcommand is used
    parser.set_defaults(func=run_align_wrapper)


def run_align_wrapper(FLAGS):
    """Wrapper to Execute the align subcommand."""
    if shutil.which("samtools") is None:
        raise RuntimeError(
            "samtools not found on PATH. Please install samtools "
            "(https://www.htslib.org) before running 'trnazap align'."
        )

    run_align(
        FLAGS.unaligned_bam,
        FLAGS.inference,
        FLAGS.out_dir,
        FLAGS.out_pre,
        FLAGS.threads,
        FLAGS.model,
        FLAGS.secondary,
        FLAGS.ident_threshold,
        FLAGS.wf_gap_open,
        FLAGS.wf_gap_extend,
        FLAGS.sw_gap_open,
        FLAGS.sw_gap_extend,
        FLAGS.sw_match,
        FLAGS.sw_mismatch,
        FLAGS.ivt_alignment,
        FLAGS.pickled_inf_obj
    )
