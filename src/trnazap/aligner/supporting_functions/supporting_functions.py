"""Utility functions used in tRNA prediction and Alignment.

This module focuses on tRNA alignment.
"""

import os
import pickle
import subprocess
import itertools

import pysam

from aligner.alignment_functions.alignment import align_read
from aligner.progress_monitoring.progress import increment_counter

from pathlib import Path


def get_model_to_ref():
    """
    Look up table from models to the associated reference and it's relative path.

    params:
        None

    return: Dictionary looking up reference from the model name
    """

    return {
        "human-mt": Path(__file__).parent.parent / 'references' / 'hg38-mature-tRNAs.fa',
        "yeast": Path(__file__).parent.parent / 'references' / 'sacCer3-mature-tRNAs.fa',
        "e_coli": Path(__file__).parent.parent / 'references' / 'eschColi_K_12_MG1655-mature-tRNAs.fa',
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

        #ref_dict[i] = {"reference_name": seq.name, "reference_seq": seq.sequence}
        ref_dict[seq.name] = {'reference_index': i, 'reference_seq':seq.sequence}
    return bam_header, ref_dict

def split_read_ids(inference_dict, threads):
    """
    Split read_ids into subsets.

    params:
        inference_dict: path to bam file to iterate through (str)
        threads: count of pools to split reads into (int)

    return: A list of sets, with each set containing read_ids
    """
    total = len(inference_dict)
    chunk_size = total // threads
    remainder = total % threads
    
    key_iter = iter(inference_dict)
    sizes = [chunk_size + (1 if i < remainder else 0) for i in range(threads)]
    return [set(itertools.islice(key_iter, size)) for size in sizes]

def make_parameter_list(
    read_id_set,
    header_dict,
    inference_dict,
    ref_dict,
    unaligned_bam_path,
    out_dir,
    out_prefix,
    all_alignments,
    monitor
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
            all_alignments,
            monitor
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


def sort_bam(args_list):
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
    bam_path, index, out_dir, out_pre, threads = args_list
    outpath = os.path.join(out_dir, f"{out_pre}_{index}.sorted.bam")
    subprocess.run(["samtools", "sort", "-@", str(threads), "-o", outpath, bam_path])
    subprocess.run(["rm", bam_path])
    return outpath

def merge_bam(bam_paths, out_dir, out_pre, threads):
    subprocess.run(["samtools",
                    "merge",
                    "-f",
                    "-@",
                    str(threads),
                    "-o",
                    os.path.join(out_dir, f"{out_pre}.sorted.bam")]+
                    bam_paths)
    
    subprocess.run(["rm"]+bam_paths)

    return None
                    

# Comments written with assistance from Anthropic Claudi Sonnet v4.0
def make_sub_bam(args_list):
    """
    Create a subset BAM file with aligned tRNA reads based on inference predictions.

    This function processes an unaligned BAM file and creates a new BAM file containing
    only the reads that were successfully classified as tRNA by the inference model.
    Each included read is aligned against its predicted reference tRNA sequence using
    the custom alignment algorithm.

    The function is designed to work in a multiprocessing context, hence the args_list
    parameter format that allows easy distribution across worker processes.

    Args:
        args_list (tuple): A tuple containing the following elements in order:
            - read_id_set (set): Set of read IDs to include in the output BAM
            - header_dict (dict): BAM header dictionary for the output file
            - inference_dict (dict): Dictionary mapping read IDs to inference results.
                Each entry should contain:
                - 'pred': Reference sequence identifier for alignment
                - 'trna_indices': Tuple of (start, end) positions for tRNA region
            - ref_dict (dict): Dictionary mapping reference IDs to reference data.
                Each entry should contain:
                - 'reference_seq': The reference sequence string for alignment
            - unaligned_bam_path (str): Path to the input unaligned BAM file
            - outpath (str): Path where the output aligned BAM file will be written

    Returns:
        str: The output file path (same as the input outpath parameter)

    Raises:
        AssertionError: If CIGAR string length doesn't match query sequence length,
            indicating an alignment inconsistency that needs investigation

    Notes:
        - Reads not present in both inference_dict and read_id_set are skipped
        - The function validates CIGAR strings to ensure alignment integrity
        - Output BAM file is automatically closed when the function completes
        - Designed for use in parallel processing workflows

    Examples:
        >>> args = (read_ids, header, inferences, references, "input.bam", "output.bam")
        >>> output_path = make_sub_bam(args)
        >>> print(f"Aligned BAM written to: {output_path}")
    """
    # Unpack the argument tuple - this format allows easy multiprocessing distribution
    (
        read_id_set,  # Set of read IDs we want to include in output
        header_dict,  # BAM header structure for output file
        inference_dict,  # Model predictions mapping read_id -> tRNA classification
        ref_dict,  # Reference sequences mapping ref_id -> sequence data
        unaligned_bam_path,  # Input BAM file with unaligned reads
        outpath,
        all_alignments,
        monitor
    ) = args_list  # Output path for the new aligned BAM file

    # Open the unaligned BAM file for reading
    # check_sq=False allows reading BAM files without proper SQ (sequence) headers
    # This is common for unaligned BAM files that may not have reference info
    ua_bam = pysam.AlignmentFile(unaligned_bam_path, check_sq=False)

    #Count worker reads processed for shared progress monitoring
    reads_processed = 0
    
    
    # Create the output BAM file with proper header information
    # Using context manager ensures file is properly closed even if errors occur
    with pysam.AlignmentFile(outpath, "w", header=header_dict) as outf:
        # Iterate through all reads in the input BAM file
        # until_eof=True ensures we read the entire file, not just aligned regions
        for read in ua_bam.fetch(until_eof=True):
            # Filter reads: only process those that meet both criteria:
            # 1. Have inference predictions (model classified them as tRNA)
            # 2. Are in our target read ID set (additional filtering criterion)
            if read.query_name not in read_id_set:
                continue

            reads_processed += 1
            if reads_processed % 5000 == 0:
                increment_counter(monitor, 5000)
            # Extract the predicted reference sequence information
            # The inference model tells us which tRNA reference this read best matches
            assigned_ref = inference_dict[read.query_name][0]
            assigned_ref_sequence = ref_dict[assigned_ref][
                "reference_seq"
            ]
            assigned_ref_index = ref_dict[assigned_ref]['reference_index']

            # Perform the actual sequence alignment using our custom algorithm
            # This creates a new AlignedSegment with proper CIGAR, coordinates, etc.
            aligned_read = align_read(
                read,
                inference_dict[read.query_name],
                assigned_ref_index,
                assigned_ref_sequence,
            )

            # Handle unmapped reads - write them as-is without further processing
            # These are reads where the alignment algorithm couldn't find a good match
            if aligned_read.is_unmapped:
                #outf.write(aligned_read)
                continue

            # Validate the cigar string <- remove for perfomance boost?
            _ = check_cigar(aligned_read.get_cigar_stats(), len(aligned_read.query_sequence), aligned_read.cigarstring, aligned_read.query_sequence, aligned_read.reference_id, aligned_read.reference_start, assigned_ref_sequence, aligned_read.get_tags())

            # Write the successfully aligned and validated read to the output BAM
            #outf.write(aligned_read)

            # Check if a secondary alignments should be performed, if all_alignments
            # Iterate and perform alignments giving a secondary mapping quality for
            # Each of the subsequent alignments. If the all_alignments flag isn't set
            # Then check for a 'gt' tag in the inference_dict. If there is a 'gt' in the
            # Inference tag perform that alignment (mostly for alignment model
            # Validation).
            '''
            if all_alignments:
                predicted_class = assigned_ref
                for ref_index in ref_dict:
                    assigned_ref = ref_dict[ref_index]["reference_name"]
                    assigned_ref_sequence = ref_dict[ref_index]["reference_seq"]
                    if predicted_class == assigned_ref:
                        continue

                    aligned_read = align_read(
                        read,
                        inference_dict[read.query_name],
                        assigned_ref,
                        assigned_ref_sequence,
                        secondary=True,
                    )

                    # Handle unmapped reads - ignore them
                    # These are reads where the alignment algorithm
                    # Couldn't find a good match.
                    # Generally caused by the signal segmenter
                    # Producing too narrow of a window
                    if aligned_read.is_unmapped:
                        continue

                    # Validate the cigar string <- remove for perfomance boost?
                    #_ = check_cigar(
                    #    aligned_read.get_cigar_stats(), len(aligned_read.query_sequence)
                    #)

                    # Write the successfully aligned and validated read
                    # To the output BAM
                    outf.write(aligned_read)
            '''
    increment_counter(monitor, len(read_id_set) - reads_processed + (reads_processed % 5000))
    #if reads_processed % 1000 != 0:
    #    increment_counter(monitor, reads_processed % 1000)
    #if reads_processed != len(read_id_set):
    #    increment_counter(monitor, reads_processed
    # Return the output path for potential chaining or confirmation
    return outpath

def check_cigar(cig_stats, query_sequence_len, cigarstring, query_sequence, reference_name, reference_start, reference_sequence, tags):
    """
    Validate read creation by asserting equality cigar and sequence lengths.

    Args:
        cig_stats: pysam formatted cigar stats
        query_sequence_len: the length of the query sequence

    Raises:
        AssertionError: If CIGAR string length doesn't match query sequence length,
            indicating an alignment inconsistency that needs investigation

    Returns:
        None
    """
    # Validate the CIGAR string integrity by checking alignment length
    # This is a critical quality control step to catch alignment bugs
    cigar_len = 0

    # Sum up all CIGAR operations that consume query sequence:
    # - Index 1: Insertions (I) - bases in query not in reference
    # - Index 4: Soft clips (S) - unaligned bases at read ends
    # - Index 7: Matches (=) - exact matches between query and reference
    # - Index 8: Mismatches (X) - substitutions between query and reference
    cigar_len += cig_stats[0][1]  # Insertions
    cigar_len += cig_stats[0][4]  # Soft clips
    cigar_len += cig_stats[0][7]  # Matches
    cigar_len += cig_stats[0][8]  # Mismatches

    # Verify that our CIGAR operations account for the entire query sequence
    # If this assertion fails, there's a bug in our alignment algorithm
    # that needs immediate attention as it indicates data corruption
    assert_fail_string = (
        "CIGAR length mismatch:" + f"{cigar_len=} {query_sequence_len=}\n {cigarstring=} {query_sequence=}\n {reference_name=} {reference_start=} \n {reference_sequence=} {tags=}"
    )
    #print(f"{cigar_len == query_sequence_len=}")
    assert cigar_len == query_sequence_len, assert_fail_string

    return None
