"""Parent Module for executing inference and alignment of tRNA.

This module contains argument parsing, overall function control,
and executes submodules.
"""

import argparse
import sys
from multiprocessing import Pool

from supporting_functions.supporting_functions import (
    load_inference_obj,
    make_parameter_list,
    make_sub_bam,
    process_ref,
    split_read_ids,
)

program_name = "tRNA_zap"
version = "05_16_25_v0.1.2"


def main(
    unaligned_bam, inference_list, out_dir, out_pre, threads, model, all_alignments
):
    """Execute tRNA basecall alignment and inference workflow.

    This function orchestrates the entire tRNA-zap workflow: it loads the appropriate
    reference based on the selected model, processes the reference to create a BAM
    header and reference lookup dictionary, loads inference data, splits read IDs
    for parallel processing, and distributes the work across multiple threads.

    Args:
        unaligned_bam (str): Path to the basecalled BAM file paired with pod5 input
            used for model inference.
        inference_list (list): List of paths to tRNA model inference result files,
            potentially from multiple sequencing runs.
        out_dir (str): Output directory path. Will attempt to create if it
        doesn't exist.
        out_pre (str): Prefix to be appended to all output files.
        threads (int): Number of processing threads to use for parallel execution.
        model (str): Target tRNA substrate model to use. Options include 'human-mt',
            'yeast', and 'e_coli'.

    Returns:
        None: Function completes without an explicit return value on success.
        None: Returns None explicitly if the selected model is not recognized.

    Note:
        The function maps the selected model to corresponding reference files,
        loads the reference, processes inference data, and distributes read alignment
        work in parallel across the specified number of threads.
    """
    # Identifying the appropriate reference based on the model selected
    model_to_ref = {
        "human-mt": "./references/human-mt_reference.fa",
        "yeast": "./references/human-mt_reference.fa",
        "e_coli": "./references/human-mt_reference.fa",
    }

    # Attempt to load model path, if the model is not recognized print a help
    # message and terminate the program.
    try:
        ref = model_to_ref[model]
    except Exception as e:
        print(f"{e}\n")
        print(
            f"{model} is not recognized, please choose from human-mt,"
            + " yeast, and e_coli."
        )
        return None

    # Construct a bam header and reference sequence lookup dict based on the
    # selected model
    bam_header, ref_dict = process_ref(
        ref, (program_name, version, program_name, sys.argv)
    )

    # Inference dict includes information for each read about the highest probablity
    # class, the indicies for tRNA in signal space, and if this is a training or
    # validation dataset it adds a ground truth label ('gt').
    inference_dict = load_inference_obj(inference_list, ref_dict)

    splt_reads = split_read_ids(unaligned_bam, threads)

    p_list = make_parameter_list(
        splt_reads,
        bam_header,
        inference_dict,
        ref_dict,
        unaligned_bam,
        out_dir,
        out_pre,
        all_alignments,
    )

    with Pool(threads) as p:
        files = p.map(make_sub_bam, p_list)

    print(files)

    # Add sorting indexing merging


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
        nargs="*",
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
        default=18,  # Get number of available threads
    )

    parser.add_argument(
        "--all_alignments",
        "-a",
        default=False,
        action="store_true",
        help="Perform alignemnts for all reference options"
        + "note this is extremely costly",
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

    FLAGS, unparsed = parser.parse_known_args()
    main(
        FLAGS.unaligned_bam,
        FLAGS.inference,
        FLAGS.out_dir,
        FLAGS.out_pre,
        FLAGS.threads,
        FLAGS.model,
        FLAGS.all_alignments,
    )
