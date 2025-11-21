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
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')
sns.set(rc={'figure.figsize': (11.7, 8.27)})

# Import all your plotting functions here
from .condition_plots import (
    plot_read_counts_per_trna,
    plot_identity_distribution,
    plot_per_trna_identity_boxen,
    plot_coverage_heatmap,
    plot_error_rate_heatmap_proportional,
    plot_alignment_length_distribution,
    plot_read_count_vs_identity,
    generate_trna_summary_table,
    plot_identity_comparison_boxen,
    plot_delta_read_percentage,
    plot_read_count_comparison_bars,
    plot_read_count_comparison_bars_tpm,
    plot_read_count_scatter_tpm,
    plot_delta_tpm_absolute,
    plot_delta_tpm_log2fc,
    plot_volcano,
    plot_per_position_error_deltas,
    generate_comparison_summary_table
)

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

def make_sub_dict(args):
    """
    Process reads for one thread, storing only passing reads organized by reference.
    """
    model, aligner, bam_path, ident_threshold, min_coverage, threads, thread_idx = args

    ref_label_dict, ref_dict, ref_lens, three_offset, five_offset = load_ref(model, aligner)
    
    # Use defaultdict to auto-initialize nested dicts
    by_ref = defaultdict(lambda: {
        'track_arrs': [],
        'identities': [],
        'alignment_lengths': [],
        'read_names': []
    })
    
    t0 = time()
    for read in pysam.AlignmentFile(bam_path).fetch(until_eof=True):

        if hash_first_hex(read.query_name) % threads != thread_idx:
            continue
        
        if read.is_unmapped or read.has_tag('pi') or len(read.query_sequence) < min_coverage:
            continue

        aligned_pairs = np.array([[x if x is not None else -1 for x in row] 
                                  for row in read.get_aligned_pairs()], dtype=np.int32)
        
        track_arr = positional_array(
            read.reference_end,
            read.reference_start,
            read.query_alignment_end,
            aligned_pairs,
            ref_dict[ref_label_dict[read.reference_name]],
            read.query_sequence,
            five_offset,
            ref_lens[ref_label_dict[read.reference_name]] - three_offset
        )
        
        if track_arr.shape[1] == 0:
            continue
        
        ident, passes, aln_length = read_pass(track_arr, True, ident_threshold, min_coverage)
        
        if not passes:
            continue
        
        # Store only passing reads
        ref = ref_label_dict[read.reference_name]
        by_ref[ref]['track_arrs'].append(track_arr)
        by_ref[ref]['identities'].append(float(ident))
        by_ref[ref]['alignment_lengths'].append(float(aln_length))
        by_ref[ref]['read_names'].append(read.query_name)
    
    # Convert lists to numpy arrays
    for ref in by_ref:
        by_ref[ref]['track_arrs'] = np.array(by_ref[ref]['track_arrs'])
        by_ref[ref]['identities'] = np.array(by_ref[ref]['identities'])
        by_ref[ref]['alignment_lengths'] = np.array(by_ref[ref]['alignment_lengths'])
    
    print(f"Thread {thread_idx}: {(time()-t0)/60:.2f} minutes to process {sum(len(d['track_arrs']) for d in by_ref.values())} passing reads")
    
    return dict(by_ref)

def multiprocess_read_dict(n_threads, bam_path, model, aligner, ident_threshold, min_coverage):
    """
    Process BAM file using multiple threads, returning organized data by tRNA reference.
    """
    args_list = [(model, aligner, bam_path, ident_threshold, min_coverage, n_threads, i) 
                 for i in range(n_threads)]
    
    with multiprocessing.Pool(n_threads) as p:
        out_dicts = p.map(make_sub_dict, args_list)

    # Merge all sub-dictionaries
    print("Merging dictionaries from all threads...")
    t0 = time()
    
    merged = defaultdict(lambda: {
        'track_arrs': [],
        'identities': [],
        'alignment_lengths': [],
        'read_names': []
    })
    
    for sub_dict in out_dicts:
        for ref, data in sub_dict.items():
            merged[ref]['track_arrs'].append(data['track_arrs'])
            merged[ref]['identities'].append(data['identities'])
            merged[ref]['alignment_lengths'].append(data['alignment_lengths'])
            merged[ref]['read_names'].extend(data['read_names'])
    
    # Concatenate arrays from all threads
    for ref in merged:
        merged[ref]['track_arrs'] = np.concatenate(merged[ref]['track_arrs'], axis=0)
        merged[ref]['identities'] = np.concatenate(merged[ref]['identities'])
        merged[ref]['alignment_lengths'] = np.concatenate(merged[ref]['alignment_lengths'])
    
    total_reads = sum(len(data['track_arrs']) for data in merged.values())
    print(f"Merged {total_reads} passing reads from {len(merged)} tRNAs in {(time()-t0)/60:.2f} minutes")
    
    return dict(merged)


# ============================================================================
# HIGH-LEVEL FUNCTIONS
# ============================================================================

def load_condition(bam_path, model, aligner='zap', n_threads=8, 
                   ident_threshold=0.75, min_coverage=25):
    """
    Load and process a single condition from a BAM file.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    model : str
        'e_coli' or 'yeast'
    aligner : str
        'zap' or 'bwa' (default: 'zap')
    n_threads : int
        Number of threads for parallel processing (default: 8)
    ident_threshold : float
        Minimum identity score for passing reads (default: 0.75)
    min_coverage : int
        Minimum alignment length for passing reads (default: 25)
    
    Returns
    -------
    dict
        Data dictionary organized by tRNA reference:
        {tRNA_name: {'track_arrs': array, 'identities': array, 
                     'alignment_lengths': array, 'read_names': list}}
    """
    print(f"\n{'='*60}")
    print(f"Loading condition: {os.path.basename(bam_path)}")
    print(f"Model: {model} | Aligner: {aligner}")
    print(f"Filters: identity >= {ident_threshold}, coverage >= {min_coverage}")
    print(f"{'='*60}\n")
    
    t0 = time()
    
    data_dict = multiprocess_read_dict(
        n_threads=n_threads,
        bam_path=bam_path,
        model=model,
        aligner=aligner,
        ident_threshold=ident_threshold,
        min_coverage=min_coverage
    )
    
    total_time = (time() - t0) / 60
    total_reads = sum(len(data['track_arrs']) for data in data_dict.values())
    
    print(f"\n{'='*60}")
    print(f"Condition loaded successfully!")
    print(f"Total passing reads: {total_reads:,}")
    print(f"Total tRNAs: {len(data_dict)}")
    print(f"Processing time: {total_time:.2f} minutes")
    print(f"{'='*60}\n")
    
    return data_dict


def one_condition_figures(single_dict, condition_label, model, 
                          out_dir, out_prefix='', ident_threshold=0.75):
    """
    Generate all single-condition figures.
    
    Parameters
    ----------
    single_dict : dict
        Data dictionary from load_condition()
    condition_label : str
        Label for this condition (e.g., "Control", "Treatment")
    model : str
        'e_coli' or 'yeast'
    out_dir : str
        Output directory for figures
    out_prefix : str
        Prefix for output filenames (default: '')
    ident_threshold : float
        Minimum identity threshold used for filtering (default: 0.75)
    
    Returns
    -------
    dict
        Dictionary of DataFrames from each plot for further analysis
    """
    print(f"\n{'='*60}")
    print(f"Generating single-condition figures: {condition_label}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # Generate summary table FIRST
    print("Generating: tRNA summary table...")
    results['summary'] = generate_trna_summary_table(
        single_dict, condition_label, model, out_dir, out_prefix
    )
    
    # 1. Read counts per tRNA
    print("Generating: Read counts per tRNA...")
    results['read_counts'] = plot_read_counts_per_trna(
        single_dict, condition_label, model, out_dir, out_prefix
    )
    
    # 2. Identity distribution
    print("Generating: Identity distribution...")
    results['identity_dist'] = plot_identity_distribution(
        single_dict, condition_label, ident_threshold, out_dir, out_prefix
    )
    
    # 3. Per-tRNA identity boxen plots
    print(f"Generating: Per-tRNA identity distributions...")
    results['identity_boxen'] = plot_per_trna_identity_boxen(
        single_dict, condition_label, model, ident_threshold, out_dir, out_prefix
    )
    
    # 4. Coverage heatmap
    print("Generating: Coverage heatmap...")
    results['coverage_heatmap'] = plot_coverage_heatmap(
        single_dict, condition_label, model, out_dir, out_prefix
    )
    
    # 5. Error rate heatmap (proportional)
    print("Generating: Error rate heatmap...")
    results['error_rate_heatmap'] = plot_error_rate_heatmap_proportional(
        single_dict, condition_label, model, out_dir, out_prefix
    )
    
    # 6. Alignment length distribution
    print("Generating: Alignment length distribution...")
    results['alignment_length'] = plot_alignment_length_distribution(
        single_dict, condition_label, out_dir, out_prefix
    )
    
    # 7. Read count vs identity scatter
    print("Generating: Read count vs identity scatter...")
    results['count_vs_identity'] = plot_read_count_vs_identity(
        single_dict, condition_label, out_dir, out_prefix
    )
    
    print(f"\n{'='*60}")
    print(f"All figures generated successfully!")
    print(f"Files saved to: {out_dir}")
    print(f"{'='*60}\n")
    
    return results


def compare_conditions(condition_a_dict, condition_a_label, 
                       condition_b_dict, condition_b_label,
                       model, out_dir, out_prefix='',
                       ident_threshold=0.75,
                       fc_threshold=1.5, pval_threshold=0.05,
                       trnas_for_position_plots=None):
    """
    Generate all comparison figures between two conditions.
    
    Parameters
    ----------
    condition_a_dict : dict
        Data dictionary for condition A from load_condition()
    condition_a_label : str
        Label for condition A (e.g., "Control")
    condition_b_dict : dict
        Data dictionary for condition B from load_condition()
    condition_b_label : str
        Label for condition B (e.g., "Treatment")
    model : str
        'e_coli' or 'yeast'
    out_dir : str
        Output directory for figures
    out_prefix : str
        Prefix for output filenames (default: '')
    ident_threshold : float
        Minimum identity threshold used for filtering (default: 0.75)
    fc_threshold : float
        Fold change threshold for volcano plot (default: 1.5)
    pval_threshold : float
        P-value threshold for volcano plot (default: 0.05)
    trnas_for_position_plots : list, optional
        List of tRNA names for per-position error delta plots
        If None, will plot for top 5 most abundant tRNAs
    
    Returns
    -------
    dict
        Dictionary of DataFrames from each plot for further analysis
    """
    print(f"\n{'='*60}")
    print(f"Comparing conditions: {condition_a_label} vs {condition_b_label}")
    print(f"Output directory: {out_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    # Generate comparison summary table FIRST
    print("Generating: Comparison summary table...")
    results['summary'] = generate_comparison_summary_table(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, fc_threshold, pval_threshold,
        out_dir, out_prefix
    )
    
    # 1. Delta read percentage
    print("Generating: Delta read percentage...")
    results['delta_percentage'] = plot_delta_read_percentage(
        condition_a_dict, condition_a_label, 
        condition_b_dict, condition_b_label,
        model, out_dir, out_prefix
    )
    
    # 2. Side-by-side read count comparison
    print("Generating: Side-by-side read count comparison...")
    results['count_comparison'] = plot_read_count_comparison_bars(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, out_dir, out_prefix
    )

    # 2b. Side-by-side tpm comparison
    print("Generating: Side-by-side TPM comparison...")
    results['tpm_comparison'] = plot_read_count_comparison_bars_tpm(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, out_dir, out_prefix
    )
    
    # 3. TPM scatter plot
    print("Generating: TPM scatter plot...")
    results['tpm_scatter'] = plot_read_count_scatter_tpm(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        out_dir, out_prefix
    )
    
    # 4. Delta TPM (absolute)
    print("Generating: Delta TPM (absolute)...")
    results['delta_tpm_absolute'] = plot_delta_tpm_absolute(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, out_dir, out_prefix
    )
    
    # 5. Delta TPM (log2FC)
    print("Generating: Delta TPM (log2 fold change)...")
    results['delta_tpm_log2fc'] = plot_delta_tpm_log2fc(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, out_dir, out_prefix
    )
    
    # 6. Volcano plot
    print("Generating: Volcano plot...")
    results['volcano'] = plot_volcano(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        fc_threshold, pval_threshold,
        out_dir, out_prefix
    )
    
    # 7. Identity comparison boxen
    print("Generating: Identity comparison boxen plot...")
    results['identity_comparison'] = plot_identity_comparison_boxen(
        condition_a_dict, condition_a_label,
        condition_b_dict, condition_b_label,
        model, ident_threshold, out_dir, out_prefix
    )
    
    # 8. Per-position error deltas
    if trnas_for_position_plots is None:
        # Get top 5 most abundant tRNAs (present in both conditions)
        common_trnas = set(condition_a_dict.keys()) & set(condition_b_dict.keys())
        trna_counts = [(trna, len(condition_a_dict[trna]['track_arrs']) + 
                       len(condition_b_dict[trna]['track_arrs'])) 
                       for trna in common_trnas]
        trna_counts.sort(key=lambda x: x[1], reverse=True)
        trnas_for_position_plots = [trna for trna, _ in trna_counts[:5]]
        print(f"Generating per-position plots for top 5 tRNAs: {trnas_for_position_plots}")
    
    results['position_deltas'] = {}
    for trna in trnas_for_position_plots:
        print(f"Generating: Per-position error deltas for {trna}...")
        results['position_deltas'][trna] = plot_per_position_error_deltas(
            condition_a_dict, condition_a_label,
            condition_b_dict, condition_b_label,
            trna, out_dir, out_prefix
        )
    
    print(f"\n{'='*60}")
    print(f"All comparison figures generated successfully!")
    print(f"Files saved to: {out_dir}")
    print(f"{'='*60}\n")
    
    return results




# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example workflow
    bam1 = sys.argv[1]
    label1 = sys.argv[2]
    bam2 = sys.argv[3]
    label2 = sys.argv[4]
    out_dir = sys.argv[7]
    
    # Set parameters
    model = sys.argv[5]
    aligner = sys.argv[6]
    n_threads = 8
    ident_threshold = 0.75
    min_coverage = 25
    
    # Load two conditions
    control = load_condition(
        bam_path=bam1,
        model=model,
        aligner=aligner,
        n_threads=n_threads,
        ident_threshold=ident_threshold,
        min_coverage=min_coverage
    )
    
    treatment = load_condition(
        bam_path=bam2,
        model=model,
        aligner=aligner,
        n_threads=n_threads,
        ident_threshold=ident_threshold,
        min_coverage=min_coverage
    )
    
    # Generate single-condition figures
    control_results = one_condition_figures(
        single_dict=control,
        condition_label=label1,
        model=model,
        out_dir=out_dir,
        out_prefix=f"{label1}_",
        ident_threshold=ident_threshold
    )
    
    treatment_results = one_condition_figures(
        single_dict=treatment,
        condition_label=label2,
        model=model,
        out_dir=out_dir,
        out_prefix=f"{label2}_",
        ident_threshold=ident_threshold
    )
    
    # Generate comparison figures
    comparison_results = compare_conditions(
        condition_a_dict=control,
        condition_a_label=label1,
        condition_b_dict=treatment,
        condition_b_label=label2,
        model=model,
        out_dir=out_dir,
        out_prefix=f"{label1}_vs_{label2}_",
        ident_threshold=ident_threshold,
        fc_threshold=1.5,
        pval_threshold=0.05
    )