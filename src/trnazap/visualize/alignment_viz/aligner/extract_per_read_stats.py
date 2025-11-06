"""
extract_per_read_stats.py
=========================
Extract per-read alignment statistics from BAM files.
Outputs TSV files with identity and error counts for each read.
"""

import numpy as np
import pysam
from collections import defaultdict
from importlib.resources import files
import json
import multiprocessing
import pandas as pd
import os
from time import time

# Import from existing modules
from .process_read import positional_array, read_pass


def hash_first_hex(read_id):
    try:
        return int(read_id[:8], 16)
    except:
        return hash(read_id) & 0xFFFFFFFF


def process_bam_chunk(args):
    """
    Process one chunk of BAM file and extract per-read statistics.
    No filtering - extracts stats for ALL aligned reads.
    """
    (bam_path, ref_dict, ref_lens, five_offset, three_offset,
     threads, thread_idx) = args
    
    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as f:
        ref_label_dict = json.load(f)
    
    read_stats = []
    
    with pysam.AlignmentFile(bam_path) as bam:
        for read in bam.fetch(until_eof=True):
            # Hash-based distribution
            if hash_first_hex(read.query_name) % threads != thread_idx:
                continue
            
            # Skip secondary/supplementary
            if read.is_secondary or read.is_supplementary:
                continue
            
            # Skip unmapped
            if read.is_unmapped:
                continue
            
            # Skip pi tag
            if read.has_tag('pi'):
                continue
            
            # Get tRNA name
            trna_name = ref_label_dict[read.reference_name]
            
            # Prepare aligned_pairs
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
            
            # Skip if no overlap
            if track_arr.shape[1] == 0:
                continue
            
            # Calculate identity (using same logic as pipeline)
            ident, passes, aln_length = read_pass(
                track_arr, 
                include_insertions=True, 
                ident_threshold=0.0,  # No threshold - get all reads
                min_coverage=0        # No minimum - get all reads
            )
            
            # Calculate error counts
            matches = int(np.nansum(track_arr[0, :]))
            insertions = int(np.nansum(track_arr[1, :]))
            coverage = int(np.nansum(track_arr[2, :]))
            deletions = int(np.nansum(track_arr[3, :]))
            mismatches = int(coverage - matches - deletions)
            
            read_stats.append({
                'read_id': read.query_name,
                'trna': trna_name,
                'identity': float(ident),
                'matches': matches,
                'mismatches': mismatches,
                'insertions': insertions,
                'deletions': deletions,
                'alignment_length': int(aln_length),
                'mapq': read.mapping_quality
            })
    
    return read_stats


def extract_per_read_stats(bam_path, ref_dict, ref_lens, five_offset, three_offset,
                          threads=8):
    """
    Extract per-read statistics from BAM file.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    ref_dict : dict
        Reference sequences by tRNA name
    ref_lens : dict
        Reference lengths by tRNA name
    five_offset : int
        5' offset (36 for BWA, 0 for Zap)
    three_offset : int
        3' offset (42 for BWA, 0 for Zap)
    threads : int
        Number of threads
        
    Returns
    -------
    pd.DataFrame
        Per-read statistics
    """
    args_list = [(bam_path, ref_dict, ref_lens, five_offset, three_offset,
                  threads, i) 
                 for i in range(threads)]
    
    with multiprocessing.Pool(threads) as p:
        sub_results = p.map(process_bam_chunk, args_list)
    
    # Flatten list of lists
    all_stats = []
    for sub_result in sub_results:
        all_stats.extend(sub_result)
    
    return pd.DataFrame(all_stats)


def load_references(model):
    """Load reference sequences from package."""
    refs = files('trnazap').joinpath('references')
    
    if model == 'e_coli':
        bwa_ref = str(refs / 'bwa_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_bwa_subset.biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_zap_ref.fa')
    elif model == 'yeast':
        bwa_ref = str(refs / 'bwa_align_references' / 'sacCer3-mature-tRNAs_bwa_subset_biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'sacCer3-mature-tRNAs_zap_ref.fa')
    else:
        raise ValueError(f"Unknown model: {model}")
    
    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as infile:
        ref_label_dict = json.load(infile)
    
    # Load BWA references
    bwa_ref_dict = {}
    bwa_ref_lens = {}
    for seq in pysam.FastxFile(bwa_ref):
        label = ref_label_dict[seq.name]
        bwa_ref_dict[label] = seq.sequence
        bwa_ref_lens[label] = len(seq.sequence)
    
    # Load Zap references
    zap_ref_dict = {}
    zap_ref_lens = {}
    for seq in pysam.FastxFile(zap_ref):
        label = ref_label_dict[seq.name]
        zap_ref_dict[label] = seq.sequence
        zap_ref_lens[label] = len(seq.sequence)
    
    return bwa_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens


def extract_stats_from_bams(bwa_bam, zap_bam, model, out_dir, out_prefix='', 
                            threads=8):
    """
    Main function to extract per-read statistics from both BAM files.
    
    Parameters
    ----------
    bwa_bam : str
        Path to BWA BAM file
    zap_bam : str
        Path to Zap BAM file
    model : str
        'e_coli' or 'yeast'
    out_dir : str
        Output directory
    out_prefix : str
        Output file prefix
    threads : int
        Number of threads for parallel processing
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("PER-READ STATISTICS EXTRACTION")
    print("="*70)
    print(f"Model: {model}")
    print(f"Threads: {threads}")
    
    # Load references
    print("\nLoading references...")
    t0 = time()
    bwa_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens = load_references(model)
    print(f"  Loaded {len(bwa_ref_dict)} BWA references")
    print(f"  Loaded {len(zap_ref_dict)} Zap references")
    print(f"  Time: {(time() - t0):.2f} seconds")
    # Process Zap BAM
    print("\n" + "-"*70)
    print("Processing Zap BAM...")
    print("-"*70)
    t0 = time()
    zap_df = extract_per_read_stats(
        bam_path=zap_bam,
        ref_dict=zap_ref_dict,
        ref_lens=zap_ref_lens,
        five_offset=0,
        three_offset=0,
        threads=threads
    )
    print(f"  Extracted stats for {len(zap_df):,} reads")
    print(f"  Time: {(time() - t0) / 60:.2f} minutes")
    
    # Save Zap
    zap_out_path = os.path.join(out_dir, f'{out_prefix}zap_per_read_stats.tsv')
    zap_df.to_csv(zap_out_path, sep='\t', index=False, float_format='%.6f')
    print(f"  Saved: {zap_out_path}")
    # Process BWA BAM
    print("\n" + "-"*70)
    print("Processing BWA BAM...")
    print("-"*70)
    t0 = time()
    bwa_df = extract_per_read_stats(
        bam_path=bwa_bam,
        ref_dict=bwa_ref_dict,
        ref_lens=bwa_ref_lens,
        five_offset=36,
        three_offset=42,
        threads=threads
    )
    print(f"  Extracted stats for {len(bwa_df):,} reads")
    print(f"  Time: {(time() - t0) / 60:.2f} minutes")
    
    # Save BWA
    bwa_out_path = os.path.join(out_dir, f'{out_prefix}bwa_per_read_stats.tsv')
    bwa_df.to_csv(bwa_out_path, sep='\t', index=False, float_format='%.6f')
    print(f"  Saved: {bwa_out_path}")
    
    
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBWA Statistics:")
    print(f"  Total reads: {len(bwa_df):,}")
    print(f"  Mean identity: {bwa_df['identity'].mean():.4f}")
    print(f"  Median identity: {bwa_df['identity'].median():.4f}")
    print(f"  Mean matches: {bwa_df['matches'].mean():.1f}")
    print(f"  Mean mismatches: {bwa_df['mismatches'].mean():.1f}")
    print(f"  Mean insertions: {bwa_df['insertions'].mean():.1f}")
    print(f"  Mean deletions: {bwa_df['deletions'].mean():.1f}")
    
    print(f"\nZap Statistics:")
    print(f"  Total reads: {len(zap_df):,}")
    print(f"  Mean identity: {zap_df['identity'].mean():.4f}")
    print(f"  Median identity: {zap_df['identity'].median():.4f}")
    print(f"  Mean matches: {zap_df['matches'].mean():.1f}")
    print(f"  Mean mismatches: {zap_df['mismatches'].mean():.1f}")
    print(f"  Mean insertions: {zap_df['insertions'].mean():.1f}")
    print(f"  Mean deletions: {zap_df['deletions'].mean():.1f}")
    
    print("\n" + "="*70)
    print("✓ Extraction complete!")
    print("="*70)
    
    return bwa_df, zap_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python extract_per_read_stats.py <bwa_bam> <zap_bam> <model> <out_dir> [out_prefix] [threads]")
        print("\nmodel: 'e_coli' or 'yeast'")
        print("\nExample:")
        print("  python extract_per_read_stats.py \\")
        print("    bwa.bam zap.bam yeast output_dir/ sample1_ 8")
        sys.exit(1)
    
    bwa_bam = sys.argv[1]
    zap_bam = sys.argv[2]
    model = sys.argv[3]
    out_dir = sys.argv[4]
    out_prefix = sys.argv[5] if len(sys.argv) > 5 else ''
    threads = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    
    bwa_df, zap_df = extract_stats_from_bams(
        bwa_bam=bwa_bam,
        zap_bam=zap_bam,
        model=model,
        out_dir=out_dir,
        out_prefix=out_prefix,
        threads=threads
    )