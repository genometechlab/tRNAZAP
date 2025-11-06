"""
Data loading and comparison functions for aligner comparison.
"""
import numpy as np
import pysam
from collections import defaultdict
from importlib.resources import files
import json
import multiprocessing

# Import from your existing module
from .process_read import positional_array, read_pass


def hash_first_hex(read_id):
    try:
        return int(read_id[:8], 16)
    except:
        return hash(read_id) & 0xFFFFFFFF


def load_alignments(bam_path, ref_dict, ref_lens, five_offset, three_offset,
                    model, threads, ident_threshold=0.75, min_coverage=25):
    """
    Load alignment data efficiently - works for both BWA and Zap.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    ref_dict : dict
        Reference sequences by tRNA name
    ref_lens : dict
        Reference lengths by tRNA name
    five_offset : int
        5' offset to trim (36 for BWA, 0 for Zap)
    three_offset : int
        3' offset to trim (42 for BWA, 0 for Zap)
    model : str
        'e_coli' or 'yeast'
    threads : int
        Number of threads for parallel processing
    ident_threshold : float
        Minimum identity threshold
    min_coverage : int
        Minimum alignment length
        
    Returns
    -------
    dict with keys:
        - 'by_trna': dict[trna_name] -> {track_arrs, identities, alignment_lengths, read_names}
        - 'by_read': dict[read_id] -> {'trna': str, 'identity': float, 'length': float}
        - 'failed_reads': set of read IDs that didn't pass identity/coverage thresholds
        - 'unmapped_reads': set of read IDs that didn't map
    """
    args_list = [(bam_path, ref_dict, ref_lens, five_offset, three_offset,
                  ident_threshold, min_coverage, threads, i) 
                 for i in range(threads)]
    
    with multiprocessing.Pool(threads) as p:
        sub_results = p.map(_process_chunk, args_list)
    
    # Merge by_trna (aggregated, small)
    by_trna = _merge_by_trna_dicts([r['by_trna'] for r in sub_results])
    
    # Merge by_read (keep as dict, not DataFrame!)
    by_read = {}
    for result in sub_results:
        by_read.update(result['by_read'])
    
    # Merge sets
    failed_reads = set()
    unmapped_reads = set()
    for result in sub_results:
        failed_reads.update(result['failed'])
        unmapped_reads.update(result['unmapped'])
    
    return {
        'by_trna': by_trna,
        'by_read': by_read,
        'failed_reads': failed_reads,
        'unmapped_reads': unmapped_reads
    }


def _process_chunk(args):
    """
    Process one chunk of alignments - unified for both BWA and Zap.
    Uses the exact positional_array and read_pass from process_read.py
    Identity and coverage are the ONLY filters.
    """
    (bam_path, ref_dict, ref_lens, five_offset, three_offset,
     ident_threshold, min_coverage, threads, thread_idx) = args
    
    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as f:
        ref_label_dict = json.load(f)
    
    by_trna = defaultdict(lambda: {
        'track_arrs': [],
        'identities': [],
        'alignment_lengths': [],
        'read_names': []
    })
    
    by_read = {}
    failed = set()
    unmapped = set()
    
    with pysam.AlignmentFile(bam_path) as bam:
        for read in bam.fetch(until_eof=True):
            # Hash-based distribution across threads
            if hash_first_hex(read.query_name) % threads != thread_idx:
                continue
            
            # Skip secondary/supplementary alignments
            if read.is_secondary or read.is_supplementary:
                continue
            
            # Handle unmapped
            if read.is_unmapped:
                unmapped.add(read.query_name)
                continue
            
            # Skip primary/secondary indicator
            if read.has_tag('pi'):
                continue
            
            # Check minimum read length
            if len(read.query_sequence) < min_coverage:
                failed.add(read.query_name)
                continue
            
            # Get tRNA name
            trna_name = ref_label_dict[read.reference_name]
            
            # Prepare aligned_pairs as numpy array with -1 for None
            aligned_pairs = np.array(
                [[x if x is not None else -1 for x in row] 
                 for row in read.get_aligned_pairs()], 
                dtype=np.int32
            )
            
            # Calculate region boundaries
            region_start = five_offset
            region_end = ref_lens[trna_name] - three_offset
            
            # Call positional_array
            track_arr = positional_array(
                read.reference_end,
                read.reference_start,
                read.query_alignment_end,
                aligned_pairs,
                ref_dict[trna_name],
                read.query_sequence,
                region_start,
                region_end
            )
            
            # Check if alignment overlaps region
            if track_arr.shape[1] == 0:
                failed.add(read.query_name)
                continue
            
            # Call read_pass - IDENTITY IS THE ONLY FILTER
            ident, passes, aln_length = read_pass(
                track_arr, 
                include_insertions=True, 
                ident_threshold=ident_threshold, 
                min_coverage=min_coverage
            )
            
            if passes:
                # Store in both structures
                by_trna[trna_name]['track_arrs'].append(track_arr)
                by_trna[trna_name]['identities'].append(float(ident))
                by_trna[trna_name]['alignment_lengths'].append(float(aln_length))
                by_trna[trna_name]['read_names'].append(read.query_name)
                
                by_read[read.query_name] = {
                    'trna': trna_name,
                    'identity': float(ident),
                    'length': float(aln_length)
                }
            else:
                failed.add(read.query_name)
    
    # Convert to numpy arrays for by_trna
    for trna in by_trna:
        by_trna[trna]['track_arrs'] = np.array(by_trna[trna]['track_arrs'])
        by_trna[trna]['identities'] = np.array(by_trna[trna]['identities'])
        by_trna[trna]['alignment_lengths'] = np.array(by_trna[trna]['alignment_lengths'])
    
    return {
        'by_trna': dict(by_trna),
        'by_read': by_read,
        'failed': failed,
        'unmapped': unmapped
    }


def _merge_by_trna_dicts(sub_dicts):
    """Merge by_trna dictionaries from multiple processes."""
    merged = {}
    
    for sub_dict in sub_dicts:
        for trna, data in sub_dict.items():
            if trna not in merged:
                merged[trna] = {
                    'track_arrs': [],
                    'identities': [],
                    'alignment_lengths': [],
                    'read_names': []
                }
            
            merged[trna]['track_arrs'].append(data['track_arrs'])
            merged[trna]['identities'].extend(data['identities'])
            merged[trna]['alignment_lengths'].extend(data['alignment_lengths'])
            merged[trna]['read_names'].extend(data['read_names'])
    
    # Concatenate arrays
    for trna in merged:
        if len(merged[trna]['track_arrs']) > 0:
            merged[trna]['track_arrs'] = np.concatenate(merged[trna]['track_arrs'], axis=0)
        else:
            merged[trna]['track_arrs'] = np.array([])
        merged[trna]['identities'] = np.array(merged[trna]['identities'])
        merged[trna]['alignment_lengths'] = np.array(merged[trna]['alignment_lengths'])
    
    return merged


def compare_alignments_lightweight(bwa_data, zap_data):
    """
    Compare alignments efficiently using dictionary lookups.
    Includes reads aligned by only one aligner (marked as "Unmapped" for the other).
    
    Parameters
    ----------
    bwa_data : dict
        Output from load_alignments (BWA)
    zap_data : dict
        Output from load_alignments (Zap)
        
    Returns
    -------
    dict with:
        - 'read_comparison': dict[read_id] -> comparison info (includes one-aligner-only reads)
        - 'stats': summary statistics dict
        - 'read_sets': dict of useful read ID sets
    """
    bwa_by_read = bwa_data['by_read']
    zap_by_read = zap_data['by_read']
    
    # Get read ID sets (fast set operations)
    bwa_reads = set(bwa_by_read.keys())
    zap_reads = set(zap_by_read.keys())
    
    both_pass = bwa_reads & zap_reads
    bwa_only = bwa_reads - zap_reads
    zap_only = zap_reads - bwa_reads
    
    # Store comparison for ALL reads that at least one aligner aligned
    read_comparison = {}
    agreements = 0
    disagreements = 0
    identity_deltas = []
    
    # Reads aligned by BOTH aligners
    for read_id in both_pass:
        bwa_info = bwa_by_read[read_id]
        zap_info = zap_by_read[read_id]
        
        agree = bwa_info['trna'] == zap_info['trna']
        if agree:
            agreements += 1
        else:
            disagreements += 1
        
        delta = zap_info['identity'] - bwa_info['identity']
        identity_deltas.append(delta)
        
        read_comparison[read_id] = {
            'bwa_trna': bwa_info['trna'],
            'zap_trna': zap_info['trna'],
            'bwa_identity': bwa_info['identity'],
            'zap_identity': zap_info['identity'],
            'bwa_length': bwa_info['length'],
            'zap_length': zap_info['length'],
            'agree': agree,
            'identity_delta': delta
        }
    
    # Reads aligned ONLY by BWA (Zap failed/unmapped)
    for read_id in bwa_only:
        bwa_info = bwa_by_read[read_id]
        read_comparison[read_id] = {
            'bwa_trna': bwa_info['trna'],
            'zap_trna': 'Unmapped',
            'bwa_identity': bwa_info['identity'],
            'zap_identity': None,
            'bwa_length': bwa_info['length'],
            'zap_length': None,
            'agree': False,
            'identity_delta': None
        }
    
    # Reads aligned ONLY by Zap (BWA failed/unmapped)
    for read_id in zap_only:
        zap_info = zap_by_read[read_id]
        read_comparison[read_id] = {
            'bwa_trna': 'Unmapped',
            'zap_trna': zap_info['trna'],
            'bwa_identity': None,
            'zap_identity': zap_info['identity'],
            'bwa_length': None,
            'zap_length': zap_info['length'],
            'agree': False,
            'identity_delta': None
        }
    
    # Summary statistics (only for reads in both)
    stats = {
        'total_bwa_aligned': len(bwa_reads),
        'total_zap_aligned': len(zap_reads),
        'both_aligned': len(both_pass),
        'bwa_only': len(bwa_only),
        'zap_only': len(zap_only),
        'agreements': agreements,
        'disagreements': disagreements,
        'agreement_rate': agreements / len(both_pass) if both_pass else 0,
        'mean_identity_delta': np.mean(identity_deltas) if identity_deltas else 0,
        'median_identity_delta': np.median(identity_deltas) if identity_deltas else 0
    }
    
    # Useful read sets
    read_sets = {
        'both_pass': both_pass,
        'bwa_only': bwa_only,
        'zap_only': zap_only,
        'agree': {rid for rid in both_pass if read_comparison[rid]['agree']},
        'disagree': {rid for rid in both_pass if not read_comparison[rid]['agree']}
    }
    
    return {
        'read_comparison': read_comparison,
        'stats': stats,
        'read_sets': read_sets
    }