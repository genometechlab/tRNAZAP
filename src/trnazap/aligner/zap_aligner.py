import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import sys
from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import time
import numba
numba.set_num_threads(1)

from .supporting_functions.supporting_functions import get_model_to_ref, process_ref, make_parameter_list, make_sort_params_list, split_read_ids, make_sub_bam, sort_bam, merge_bam
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
    wf_gap_open,
    wf_gap_extend, 
    sw_gap_open,
    sw_gap_extend,
    sw_match,
    sw_mismatch,
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

    program_name = "tRNA_zap"
    version = "05_16_25_v0.1.2"
    # Identifying the appropriate reference based on the model selected
    model_to_ref = get_model_to_ref()

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
    inference_dict = load_inference_obj(inference_list, pickled)

    #splt_reads = split_read_ids(inference_dict, threads)

    monitor_counter = create_shared_counter()
    monitor = create_monitor(monitor_counter.name, len(inference_dict))
    monitor.start()
    
    p_list = make_parameter_list(
        threads,
        bam_header,
        inference_dict,
        ref_dict,
        unaligned_bam,
        out_dir,
        out_pre,
        secondary,
        monitor_counter.name,
        wf_gap_open, 
        wf_gap_extend, 
        sw_gap_open,
        sw_gap_extend,
        sw_match,
        sw_mismatch
    )

    with Pool(threads) as p:
        files = p.map(make_sub_bam, p_list)

    print("Finished Aligning")
    # Update counter to complete the progress bar
    current_count = get_counter_value(monitor_counter.name)
    missing = len(inference_dict) - current_count  # You'll need to have total_work defined
    
    if missing > 0:
        print(f"Updating counter with {missing} missing reads")
        increment_counter(monitor_counter.name, missing)
        
        # Give monitor thread time to see the update and reach 100%
        time.sleep(0.2)
    elif missing < 0:
        print(f"WARNING: Counter exceeded expected by {-missing}")
    
    # Now check and stop the monitor
    monitor.join(timeout=5)

    if monitor.is_alive():
        print("WARNING: Monitor thread still running!")
    monitor_counter.close()
    monitor_counter.unlink()
    
    print(files)

    sort_p_list = make_sort_params_list(
        files, 
        out_dir, 
        out_pre, 
        threads)
    
    with Pool(threads) as p:
        files = p.map(sort_bam, sort_p_list)

    merge_bam(files, out_dir, out_pre, threads)