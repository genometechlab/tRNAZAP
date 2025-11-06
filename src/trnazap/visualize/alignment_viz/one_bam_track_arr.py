import numpy as np
from dataclasses import dataclass
import os
from collections import defaultdict
import warnings
import multiprocessing
import json
import sys
from .process_read import positional_array, read_pass
from importlib.resources import files
import pysam
from time import time

@dataclass
class ReadData:
    track_arr: np.ndarray
    ref_name: str
    passes: bool
    identity: float
    alignment_length: float

def load_ref(model, aligner):
    refs = files('trnazap').joinpath('references')
    if model == 'e_coli':
        bwa_ref = str(refs / 'bwa_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_bwa_subset.biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_zap_ref.fa')

    if model == 'yeast':
        bwa_ref = str(refs / 'bwa_align_references' / 'sacCer3-mature-tRNAs_bwa_subset_biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'sacCer3-mature-tRNAs_zap_ref.fa')

    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as infile:
        ref_label_dict = json.load(infile)

    if aligner == "bwa":
        ref = bwa_ref
        five_offset = 36
        three_offset = 42
    
    else:
        ref = zap_ref
        five_offset = 0
        three_offset = 0

    ref_dict = {}
    ref_lens = {}
    
    for seq in pysam.FastxFile(ref):
        ref_dict[ref_label_dict[seq.name]] = seq.sequence
        ref_lens[ref_label_dict[seq.name]] = len(seq.sequence)
        
    return ref_label_dict, ref_dict, ref_lens, three_offset, five_offset

def hash_first_hex(read_id):
    try:
        return int(read_id[:8], 16)
    except:
        return hash(read_id) & 0xFFFFFFFF
        
def multiprocess_read_dict(n_threads, bam_path, model, aligner, ident_threshold, min_coverage):

    args_list = [(model,
                  aligner,
                  bam_path,
                  ident_threshold,
                  min_coverage,
                  n_threads,
                  i
                 ) for i in range(n_threads)]
    with multiprocessing.Pool(n_threads) as p:
        out_dicts = p.map(make_sub_dict, args_list)

    t0 = time()
    read_dict = {}
    for d in out_dicts:
        read_dict.update(d)
    print(f"{(time()-t0)/60} minutes to merge dicts")

    return read_dict
                 

def make_sub_dict_old(args):

    model, aligner, bam_path, ident_threshold, min_coverage, threads, thread_idx = args

    ref_label_dict, ref_dict, ref_lens, three_offset, five_offset = load_ref(model, aligner)
    
    read_dict = {}
    t0 = time()
    for read in pysam.AlignmentFile(bam_path).fetch(until_eof=True):

        if hash_first_hex(read.query_name)%threads != thread_idx:
                continue
        
        if read.is_unmapped or read.has_tag('pi') or len(read.query_sequence) < min_coverage:
            read_dict[read.query_name] = None
            continue

        aligned_pairs = np.array([[x if x is not None else -1 for x in row] for row in read.get_aligned_pairs()], dtype=np.int32)
        track_arr = positional_array(read.reference_end,
                                     read.reference_start,
                                     read.query_alignment_end,
                                     aligned_pairs,
                                     ref_dict[ref_label_dict[read.reference_name]],
                                     read.query_sequence,
                                     five_offset,
                                     three_offset)
        ident, p, length = read_pass(track_arr, ident_threshold=ident_threshold, min_coverage=min_coverage)

        read_dict[read.query_name] = ReadData(
            track_arr=track_arr,
            ref_name=ref_label_dict[read.reference_name],
            passes=p,  # Convert numpy bool to Python bool
            identity=ident,
            alignment_length=length
        )

    print(f"{(time()-t0)/60} minutes to process {len(read_dict)} reads")

    return read_dict

def make_sub_dict(args):

    model, aligner, bam_path, ident_threshold, min_coverage, threads, thread_idx = args

    ref_label_dict, ref_dict, ref_lens, three_offset, five_offset = load_ref(model, aligner)
    
    by_ref = defaultdict(lambda: {
        'track_arrs': [],
        'identities': [],
        'alignment_lengths': [],
        'read_names': []
    })
    t0 = time()
    for read in pysam.AlignmentFile(bam_path).fetch(until_eof=True):

        if hash_first_hex(read.query_name)%threads != thread_idx:
                continue
        
        if read.is_unmapped or read.has_tag('pi') or len(read.query_sequence) < min_coverage:
            continue

        aligned_pairs = np.array([[x if x is not None else -1 for x in row] for row in read.get_aligned_pairs()], dtype=np.int32)

        track_arr = positional_array(read.reference_end,
                                     read.reference_start,
                                     read.query_alignment_end,
                                     aligned_pairs,
                                     ref_dict[ref_label_dict[read.reference_name]],
                                     read.query_sequence,
                                     five_offset,
                                     ref_lens[ref_label_dict[read.reference_name]])

        
        if track_arr.shape[1] == 0:
            continue
        
        ident, passes, aln_length = read_pass(track_arr, True, ident_threshold, min_coverage)

        if not passes:  # Skip failing reads
            continue
        
        # Store only passing reads
        ref = ref_label_dict[read.reference_name]
        by_ref[ref]['track_arrs'].append(track_arr)
        by_ref[ref]['identities'].append(float(ident))
        by_ref[ref]['alignment_lengths'].append(float(aln_length))
        by_ref[ref]['read_names'].append(read.query_name)
    
    # Convert to numpy arrays
    for ref in by_ref:
        by_ref[ref]['track_arrs'] = np.array(by_ref[ref]['track_arrs'])
        by_ref[ref]['identities'] = np.array(by_ref[ref]['identities'])
        by_ref[ref]['alignment_lengths'] = np.array(by_ref[ref]['alignment_lengths'])
    print(f"{(time()-t0)/60} minutes")
    return dict(by_ref)
        
        
if __name__ == "__main__":
    from time import time
    t0 = time()
    x = multiprocess_read_dict(
        n_threads = 8,
        bam_path = sys.argv[1],
        model = "yeast",
        aligner = "zap",
        ident_threshold = 0.75,
        min_coverage = 25)
    print(f"{(time() - t0) / 60} minutes to evaluate {len(x)} reads.")