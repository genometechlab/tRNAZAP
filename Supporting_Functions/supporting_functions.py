"""Utility functions used in tRNA prediction and Alignment.

This module focuses on tRNA alignment.
"""

import os
import pickle
import subprocess

import pysam


def get_model_to_ref():
    """
    Look up table from models to the associated reference and it's relative path.

    params:
        None

    return: Dictionary looking up reference from the model name
    """
    return {
        "human-mt": "./references/human-mt_reference.fa",
        "yeast": "./references/human-mt_reference.fa",
        "e_coli": "./references/human-mt_reference.fa",
    }


def process_ref(ref_path, program_info):
    """
    Construct a header for the bam file (SQ/PG tags).

    params:
        ref_path: Path to the reference as looked up through
        the model dictionary (str)
        program_info: Additional information for
        constructing the PG tag, internally
        handlered, no user provided information required.
        (tuple(str, str, str, str))

    returns: Bam header dict, lookup table from numeric
    tRNA key to tRNA sequence and label
    """
    # Initialize reference lookup dictionary
    ref_dict = {}

    # Set up bam header dictionary including HD PG and SQ lines
    bam_header = {"HD": {"VN": "1.0", "SO": "unsorted"}}
    program_name, program_version, program_id, command_line = program_info
    command_line = " ".join(command_line)

    bam_header["SQ"] = []

    fastx_file = pysam.FastxFile(ref_path)
    for i, seq in enumerate(fastx_file):
        # Update the SQ tags
        bam_header["SQ"].append({"LN": len(seq.sequence), "SN": seq.name})

        ref_dict[i] = {"reference_name": seq.name, "reference_seq": seq.sequence}

    return bam_header, ref_dict


def load_inference_obj(inference_path_list, ref_dict):
    """
    Read in the pickled inference object(s).

    params:
        inference_path_list: list of paths to pickled results from the
        inference step
        ref_dict: Dictionary of numeric tRNA reference code keys with sequence
        and name information.

    return: A dictionary for each read in the dataset.
    """
    inference_dict = {}
    for inference_path in inference_path_list:
        with open(inference_path, "rb") as infile:
            inf_obj = pickle.load(infile)

        for read_id in inf_obj:
            inference_dict[read_id] = {
                "pred": inf_obj[read_id]["pred"],
                "scores": inf_obj[read_id]["scores"],
            }
            if inf_obj[read_id]["trna_indices"] != (-1, -1):
                inference_dict[read_id]["trna_indices"] = (
                    64 * inf_obj[read_id]["trna_indices"][0],
                    64 * inf_obj[read_id]["trna_indices"][1],
                )
            else:
                inference_dict[read_id]["trna_indices"] = (-1, -1)
            if "gt" in inf_obj[read_id]:
                inference_dict[read_id]["gt"] = ref_dict[inf_obj[read_id]["gt"]][
                    "reference_name"
                ]

    return inference_dict


def split_read_ids(unaligned_bam, threads):
    """
    Split read_ids into subsets.

    params:
        unaligned_bam: path to bam file to iterate through (str)
        threads: count of pools to split reads into (int)

    return: A list of sets, with each set containing read_ids
    """
    read_id_sets = [set() for i in range(threads)]
    bam_fh = pysam.AlignmentFile(unaligned_bam, check_sq=False)
    for i, read in enumerate(bam_fh.fetch(until_eof=True)):
        read_id_sets[i % threads].add(read.query_name)
    return read_id_sets


def make_parameter_list(
    read_id_set,
    header_dict,
    inference_dict,
    ref_dict,
    unaligned_bam_path,
    out_dir,
    out_prefix,
):
    """
    Make a list of parameters to multiprocess alignments.

    params:
        read_id_set: a list of sets containing the read_ids to be processed for each
        sub bam
        header_dict: The header to be used for the final bam, as well as each sub_bam
        inference_dict: Dict object from the inference stage of the pipeline
        ref_dict: Dict object containing reference indexes, names, and lengths
        unaligned_bam_path: Path to the unaligned bam file (used for matching read ids
        to sequence
        and move tables)
        out_dir: User defined directory to output files
        out_prefix: User defined prefix to append to the start of files

    returns:
        A list of tuples containing the required parameters for each process being
        executed
    """
    return [
        (
            read_id_set[i],
            header_dict,
            inference_dict,
            ref_dict,
            unaligned_bam_path,
            os.path.join(out_dir, f"{out_prefix}_{i}_temporary.bam"),
        )
        for i in range(len(read_id_set))
    ]


def make_sort_params_list(unsorted_files, out_dir, out_pre, threads):
    """
    Make a list of parameters to pass to sorting of alignemnts.

    params:
        unsorted_files: List of unsorted bam paths
        out_dir: User defined directory to output files
        out_prefix: User defined prefix to append to the start of files
        threads: Thread count for each execution

    returns:
        A list of tuples containing the required parameters for each process being
        executed
    """
    return [
        (unsorted_files[i], str(i), out_dir, out_pre, threads)
        for i in range(len(unsorted_files))
    ]


def sort_bam(bam_path, index, out_dir, out_pre, threads):
    """
    Wrap function to sort bam files.

    params:
        bam_path: path to bam file to sort (str)
        index: identifier for sub_bam as a string (str)
        out_dir: output directory path (str)
        out_pre: output prefix (str)
        threads: count of pools to split reads into (int)

    return: sorted_bam_path
    """
    outpath = os.path.join(out_dir, f"{out_pre}.sorted.bam")
    subprocess.run(["samtools", "sort", "-@", str(threads), "-o", outpath])

    return outpath
