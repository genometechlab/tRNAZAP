# src/trnazap/cli/align_cli.py

import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
from multiprocessing import Pool
import time
import numba
numba.set_num_threads(1)

from ..aligner.supporting_functions.supporting_functions import (
    make_parameter_list,
    make_sub_bam,
    process_ref,
    split_read_ids,
    get_model_to_ref,
    make_sort_params_list,
    sort_bam,
    merge_bam
)

from ..aligner.progress_monitoring.progress import (
    create_shared_counter,
    increment_counter,
    create_monitor,
    get_counter_value
)

from ..aligner.zap_aligner import run_align

from ..aligner.inference_functions.process_inference import load_inference_obj

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
        default="human-mt",
        choices=["human-mt", "yeast", "e_coli"],
        help="Target substrate. Currently three models are supported human"
        + "mitochondrail tRNA (human-mt), yeast tRNA (yeast),"
        + " and E. Coli tRNA (e_coli).",
    )

    parser.add_argument(
        "--pickled_inf_obj",
        default=False,
        action="store_true",
        help="A pre pickled inference obj to reduce repeat run speed"
    )

    # Wagner-Fisher alignment parameters
    parser.add_argument("--wf_gap_open", type=float, default=2.0, 
                       help="Wagner-Fisher gap open penalty (default: 2.0)")
    parser.add_argument("--wf_gap_extend", type=float, default=0.5,
                       help="Wagner-Fisher gap extend penalty (default: 0.5)")
    
    # Smith-Waterman alignment parameters
    parser.add_argument("--sw_gap_open", type=float, default=-6.0,
                       help="Smith-Waterman gap open penalty (default: -6.0)")
    parser.add_argument("--sw_gap_extend", type=float, default=-1.0,
                       help="Smith-Waterman gap extend penalty (default: -1.0)")
    parser.add_argument("--sw_match", type=float, default=3.0,
                       help="Smith-Waterman match score (default: 3.0)")
    parser.add_argument("--sw_mismatch", type=float, default=1.0,
                       help="Smith-Waterman mismatch penalty (default: 1.0)")
    
    # Set the function to call when this subcommand is used
    parser.set_defaults(func=run_align_wrapper)


def run_align_wrapper(FLAGS):
    """Wrapper to Execute the align subcommand."""
    
    run_align(
        FLAGS.unaligned_bam,
        FLAGS.inference,
        FLAGS.out_dir,
        FLAGS.out_pre,
        FLAGS.threads,
        FLAGS.model,
        FLAGS.secondary,
        FLAGS.wf_gap_open,
        FLAGS.wf_gap_extend,
        FLAGS.sw_gap_open,
        FLAGS.sw_gap_extend,
        FLAGS.sw_match,
        FLAGS.sw_mismatch,
        FLAGS.pickled_inf_obj
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        default="human-mt",
        choices=["human-mt", "yeast", "e_coli"],
        help="Target substrate. Currently three models are supported human"
        + "mitochondrail tRNA (human-mt), yeast tRNA (yeast),"
        + " and E. Coli tRNA (e_coli).",
    )

    parser.add_argument(
        "--pickled_inf_obj",
        default=False,
        action="store_true"
    )

    # Wagner-Fisher alignment parameters
    parser.add_argument("--wf_gap_open", type=float, default=2.0, 
                       help="Wagner-Fisher gap open penalty (default: 2.0)")
    parser.add_argument("--wf_gap_extend", type=float, default=0.5,
                       help="Wagner-Fisher gap extend penalty (default: 0.5)")
    
    # Smith-Waterman alignment parameters
    parser.add_argument("--sw_gap_open", type=float, default=-6.0,
                       help="Smith-Waterman gap open penalty (default: -6.0)")
    parser.add_argument("--sw_gap_extend", type=float, default=-1.0,
                       help="Smith-Waterman gap extend penalty (default: -1.0)")
    parser.add_argument("--sw_match", type=float, default=3.0,
                       help="Smith-Waterman match score (default: 3.0)")
    parser.add_argument("--sw_mismatch", type=float, default=1.0,
                       help="Smith-Waterman mismatch penalty (default: 1.0)")

    

    FLAGS, unparsed = parser.parse_known_args()
    run_align(
        FLAGS.unaligned_bam,
        FLAGS.inference,
        FLAGS.out_dir,
        FLAGS.out_pre,
        FLAGS.threads,
        FLAGS.model,
        FLAGS.secondary,
        FLAGS.wf_gap_open,
        FLAGS.wf_gap_extend,
        FLAGS.sw_gap_open,
        FLAGS.sw_gap_extend,
        FLAGS.sw_match,
        FLAGS.sw_mismatch,
        FLAGS.pickled_inf_obj
    )