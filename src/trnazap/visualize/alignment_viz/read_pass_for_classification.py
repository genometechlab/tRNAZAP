import pysam
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
sns.set_theme(style='white')
from matplotlib.colors import LogNorm
import numpy as np
from numba import njit
from importlib.resources import files
import pickle
import time

def ref_conversion(bwa_ref, zap_ref):
    zap_seq_lookup = {}
    for seq in pysam.FastxFile(zap_ref):
        zap_seq_lookup[seq.sequence] = seq.name
        
    ref_conversion_dict = {}
    for seq in pysam.FastxFile(bwa_ref):
        trimmed_seq = seq.sequence[len("GGGTCAGTCATCATATGGAGCAAGAAGAAGCCTGGT"):-1*len("GGCTTCTTCTTGCTCCATCACGATCACTCATCAAAAAAAAAA")]
        ref_conversion_dict[seq.name] = zap_seq_lookup[trimmed_seq]
        
    return ref_conversion_dict
    
def positional_array(read, ref_seq, region_start, region_end):
    """
    """
    if read.is_unmapped:
        return None, None
    
    # Use pysam's reference_start and reference_end for efficient overlap check
    if read.reference_end <= region_start or read.reference_start >= region_end:
        return None, None   

    track_arr = np.full((3, (region_end - region_start)), np.nan)
    track_arr[1] = 0 #Insertion count?
    track_arr[2] = 0 #Covered position?
    
    # Track the most recent reference position we've seen
    last_ref_pos = None
    
    # Get aligned pairs (query_pos, ref_pos)
    for query_pos, ref_pos in read.get_aligned_pairs():   
        if query_pos is not None and query_pos > read.query_alignment_end:
            break
        if ref_pos is None:  # Insertion
            if last_ref_pos is None:
                continue
            idx = last_ref_pos - region_start
            # Only count insertion if it's within our region
            # Use the last reference position we saw to determine this
            if last_ref_pos is not None and region_start <= last_ref_pos < region_end:
                track_arr[1][idx] += 1
            continue
        last_ref_pos = ref_pos
        if ref_pos < region_start or ref_pos >= region_end:
            continue

        idx = ref_pos - region_start
        if np.isnan(track_arr[0][idx]):
            track_arr[0][idx] = 0
            
        if query_pos is None:  # Deletion
            continue
        else:
            # Check if it's a match or mismatch
            query_base = read.query_sequence[query_pos]
            ref_base = ref_seq[ref_pos]
            
            if query_base == ref_base:
                track_arr[0][idx] += 1
            else:
                continue
    track_arr[2] = (~np.isnan(track_arr[0])).astype(np.float64)
    return track_arr
    
@njit()
def read_pass(track_arr, include_insertions = False, ident_threshold = 0.70, min_coverage = 15):

    total = np.count_nonzero(~np.isnan(track_arr[0]))
    if include_insertions:
        total += np.nansum(track_arr[1])
    matches = np.nansum(track_arr[0])
    
    ident = matches / total

    return ident, (ident >= ident_threshold) & (np.nansum(track_arr[2]) >= min_coverage)

def hash_first_hex(read_id):
    try:
        return int(read_id[:8], 16)
    except:
        return hash(read_id) & 0xFFFFFFFF

def multiprocess_trna_data(args):
    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.pkl'), 'rb') as infile:
        ref_label_dict = pickle.load(infile)
    bwamem, zap, bwa_mem_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens, threads, thread_idx = args
    ref_set = {'Unmapped'}
    threads=int(threads)
    read_dict = defaultdict(trna_factory)
    read_ident_dict = {}
    bwa_tRNA_dict = {}
    bwa_ident_array = defaultdict(list)
    exclusion_id_set = set()

    zap_to_bwa = {}
    for ref in zap_ref_dict:
        if "mito" in ref:
            zap_to_bwa[ref] = ref

        else:
            split_ref = ref.split('-')
            split_ref[-2] = "1"
            if split_ref[-1] != "1":
                continue
            assert split_ref[-1] == "1", print(split_ref)
            zap_to_bwa[ref] = "-".join(split_ref)
            assert zap_to_bwa[ref][-3:] == "1-1"
    
    with pysam.AlignmentFile(bwamem) as infile:
        for read in tqdm(infile, position=thread_idx*2, leave=True, total=None):
            if hash_first_hex(read.query_name)%threads != thread_idx:
                continue
            if read.has_tag('pi'):
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_unmapped:
                read_dict[read.query_name]['bwa'] = 'Unmapped'
                continue
            if read.mapping_quality <= 0:
                exclusion_id_set.add(read.query_name)
                continue

            ref_set.add(ref_label_dict[read.reference_name])
            track_arr = positional_array(read, 
                                         bwa_mem_ref_dict[ref_label_dict[read.reference_name]], 
                                         36, 
                                         bwa_ref_lens[ref_label_dict[read.reference_name]] - 43)
            if track_arr[0] is None:
                read_dict[read.query_name]['bwa'] = 'No tRNA'
                continue
                
            ident, p = read_pass(track_arr, include_insertions=True, min_coverage = 15)
            read_ident_dict[read.query_name] = {'bwa':ident, 'zap':None}
            
            if p:
                bwa_ident_array[ref_label_dict[read.reference_name]].append(ident)
                if ref_label_dict[read.reference_name] not in bwa_tRNA_dict:
                    bwa_tRNA_dict[ref_label_dict[read.reference_name]] = np.full((3, track_arr.shape[1]), 0)
                bwa_tRNA_dict[ref_label_dict[read.reference_name]] = increment_array(bwa_tRNA_dict[ref_label_dict[read.reference_name]], track_arr)
                read_dict[read.query_name]['bwa']=ref_label_dict[read.reference_name]
            else:
                read_dict[read.query_name]['bwa'] = 'Failed'
    
    #This is for the zap
    zap_tRNA_dict = {}
    zap_ident_array = defaultdict(list)
    with pysam.AlignmentFile(zap) as infile:
        time.sleep(0.01*thread_idx)
        for read in tqdm(infile, position=thread_idx*2 + 1, leave=True, total=None):
            if hash_first_hex(read.query_name)%threads != thread_idx:
                continue
            if read.query_name in exclusion_id_set:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_unmapped:
                read_dict[read.query_name]['zap'] = 'Unmapped'
                read_dict[read.query_name]['zap_to_bwa'] = 'Unmapped'
                continue
            ref_set.add(ref_label_dict[read.reference_name])
            track_arr = positional_array(read, 
                                         zap_ref_dict[ref_label_dict[read.reference_name]], 
                                         0, 
                                         zap_ref_lens[ref_label_dict[read.reference_name]]-1
                                         )
            if track_arr[0] is None:
                read_dict[read.query_name]['zap'] = 'No tRNA'
                read_dict[read.query_name]['zap_to_bwa'] = 'Unmapped'
                continue
                
            ident, p = read_pass(track_arr, include_insertions=True, min_coverage = 15)
                
            if read.query_name not in read_ident_dict:
                read_ident_dict[read.query_name] = {'bwa':None, 'zap':ident}
            else:
                read_ident_dict[read.query_name]['zap'] = ident
            if p:
                zap_ident_array[ref_label_dict[read.reference_name]].append(ident)
                if ref_label_dict[read.reference_name] not in zap_tRNA_dict:
                    zap_tRNA_dict[ref_label_dict[read.reference_name]] = np.full((3, track_arr.shape[1]), 0)
                zap_tRNA_dict[ref_label_dict[read.reference_name]] = increment_array(zap_tRNA_dict[ref_label_dict[read.reference_name]], track_arr)
                read_dict[read.query_name]['zap'] = ref_label_dict[read.reference_name]
                read_dict[read.query_name]['zap_to_bwa'] = zap_to_bwa[ref_label_dict[read.reference_name]]
            else:
                read_dict[read.query_name]['zap'] = 'Failed'
                read_dict[read.query_name]['zap_to_bwa'] = 'Failed'

    return ref_set, read_dict, bwa_ident_array, bwa_tRNA_dict, zap_ident_array, zap_tRNA_dict, read_ident_dict, exclusion_id_set

def merge_multiprocess(outputs):
    
    ref_set = outputs[0][0].union(*[outputs[i][0] for i in range(len(outputs))])

    total_read_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][1]:
            assert key not in total_read_dict
            total_read_dict[key] = outputs[i][1][key]
    
    bwa_ident_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][2]:
            if key not in bwa_ident_dict:
                bwa_ident_dict[key] = []
            bwa_ident_dict[key].extend(outputs[i][2][key])

    zap_ident_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][4]:
            if key not in zap_ident_dict:
                zap_ident_dict[key] = []
            zap_ident_dict[key].extend(outputs[i][4][key])

    bwa_position_ident_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][3]:
            if key not in bwa_position_ident_dict:
                bwa_position_ident_dict[key] = outputs[i][3][key]
            else:
                bwa_position_ident_dict[key] = increment_array(bwa_position_ident_dict[key], outputs[i][3][key])

    zap_position_ident_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][5]:
            if key not in zap_position_ident_dict:
                zap_position_ident_dict[key] = outputs[i][5][key]
            else:
                zap_position_ident_dict[key] = increment_array(zap_position_ident_dict[key], outputs[i][5][key])

    read_ident_dict = {}
    for i in range(len(outputs)):
        for key in outputs[i][6]:
            read_ident_dict[key] = outputs[i][6][key]

    exclusion_read_ids = set()
    for i in range(len(outputs)):
        for read_id in outputs[i][7]:
            exclusion_read_ids.add(read_id)

    return ref_set, total_read_dict, bwa_ident_dict, zap_ident_dict, bwa_position_ident_dict, zap_position_ident_dict, read_ident_dict, exclusion_read_ids

#@njit(fastmath = True) 
@njit
def increment_array(template_arr, new_arr):
    for i in range(new_arr.shape[0]):
        for j in range(new_arr.shape[1]):
            if np.isnan(new_arr[i, j]):
                continue
            template_arr[i, j] += new_arr[i, j]
    return template_arr

def trna_factory():
    return {'zap':'Unmapped', 'zap_to_bwa':'Unmapped', 'bwa':'Unmapped'}