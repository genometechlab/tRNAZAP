import os
os.environ['OMP_DISPLAY_ENV'] = 'FALSE'
os.environ['KMP_WARNINGS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import time
import numba
numba.set_num_threads(1)

from .supporting_functions.supporting_functions import get_model_to_ref, process_ref, make_parameter_list, make_sort_params_list, make_sub_bam, sort_bam, merge_bam
from .alignment_functions.alignment_defaults import default_align_params, format_align_params, profile_name, validate_align_params
from .inference_functions.process_inference import load_inference_obj
from .progress_monitoring.progress import create_shared_counter, create_monitor, get_counter_value, increment_counter

def run_align(
    unaligned_bam,
    inference_list,
    out_dir,
    out_pre,
    threads,
    model,
    secondary,
    ident_threshold = None,
    wf_gap_open = None,
    wf_gap_extend = None,
    sw_gap_open = None,
    sw_gap_extend = None,
    sw_match = None,
    sw_mismatch = None,
    ivt_alignment = False,
    pickled = False
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
        model (str): Target tRNA substrate. Either 'yeast' or 'e_coli'. Selects
            both the reference and the default scoring profile.
        secondary (bool): Also align the second highest classification and keep
            the better of the two alignments.
        ident_threshold, wf_gap_open, wf_gap_extend, sw_gap_open, sw_gap_extend,
            sw_match, sw_mismatch (float or None): Scoring parameters. None takes
            the value from the selected profile, anything else overrides it.
        ivt_alignment (bool): Use the in vitro transcribed scoring profile for
            the substrate rather than the biological one.
        pickled (bool): Treat the inference input as a pre-pickled object.

    Returns:
        None: Function completes without an explicit return value on success.

    Raises:
        ValueError: If the selected model is not a recognized substrate.

    Note:
        The function maps the selected model to corresponding reference files,
        loads the reference, processes inference data, and distributes read alignment
        work in parallel across the specified number of threads.
    """

    program_name = "tRNA_zap"
    version = "05_16_25_v0.1.2"

    #If out dir directory does not exist, will try and create.
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Resolve the scoring parameters. Anything the caller passed explicitly
    # overrides the substrate's profile; everything else comes from the profile.
    # This raises if the model is not a recognized substrate, so it also guards
    # the reference lookup below.
    align_profile = profile_name(model, ivt_alignment)
    params = default_align_params(
        model,
        ivt_alignment,
        ident_threshold = ident_threshold,
        wf_gap_open = wf_gap_open,
        wf_gap_extend = wf_gap_extend,
        sw_gap_open = sw_gap_open,
        sw_gap_extend = sw_gap_extend,
        sw_match = sw_match,
        sw_mismatch = sw_mismatch,
    )
    validate_align_params(params)

    # Scoring defaults are implicit now that they come from a profile, so the
    # run log is the only record of what an alignment actually used.
    print(f"Alignment profile: {align_profile}")
    print(f"Alignment parameters: {format_align_params(params)}")

    # Identifying the appropriate reference based on the model selected
    ref = get_model_to_ref()[model]

    # Construct a bam header and reference sequence lookup dict based on the
    # selected model. The profile and resolved parameters go into the PG tag so
    # the output bam records which scoring values produced it.
    bam_header, ref_dict = process_ref(
        ref,
        (
            program_name,
            version,
            program_name,
            sys.argv,
            f"profile={align_profile} {format_align_params(params)}",
        ),
    )

    # Inference dict includes information for each read about the highest probablity
    # class, the indicies for tRNA in signal space, and if this is a training or
    # validation dataset it adds a ground truth label ('gt').
    #inference_dict = load_inference_obj(inference_list, pickled)
    
    p_list = make_parameter_list(
        threads,
        bam_header,
        inference_list,
        ref_dict,
        unaligned_bam,
        out_dir,
        out_pre,
        secondary,
        None,
        params
    )

    # multiprocessing.Pool.map waits forever if a worker dies abruptly (OOM kill,
    # segfault) -- the run just stops with no traceback. ProcessPoolExecutor
    # raises BrokenProcessPool instead. Verified equivalent otherwise: same
    # results in the same order, same exception propagation, and it honors the
    # 'fork' start method set above so workers still inherit the parent's imports.
    # Note ex.map returns a lazy iterator, hence the list().
    #with Pool(threads) as p:
    #    files = p.map(make_sub_bam, p_list)
    with ProcessPoolExecutor(max_workers=threads) as ex:
        files = list(ex.map(make_sub_bam, p_list))

    print("Finished Aligning")

    sort_p_list = make_sort_params_list(
        files, 
        out_dir, 
        out_pre, 
        threads)
    
    with Pool(threads) as p:
        files = p.map(sort_bam, sort_p_list)

    merge_bam(files, out_dir, out_pre, threads)