"""
Main wrapper for aligner comparison analysis.
"""
import os
import pysam
import json
from importlib.resources import files
from time import time

from .load_alignments import load_alignments, compare_alignments_lightweight
from .comparison_plots import (
    suppress_plotting_warnings,
    load_sort_order,
    plot_per_class_identity_boxen,
    plot_class_counts,
    plot_class_count_deltas,
    plot_per_position_error_comparison_heatmap,
    plot_alignment_classification_heatmap,
    run_statistical_comparisons,
    plot_length_identity_heatmaps,
    plot_per_read_identity_2dhist,
    plot_misclassified_identity_2dhist,
    plot_one_aligner_only_histograms,
    create_summary_statistics_table,
    plot_identity_histograms,
    plot_positional_error_barplots
)


def load_references(model, reference=None):
    """
    Load BWA and Zap reference sequences from package based on model.
    
    Parameters
    ----------
    model : str
        'e_coli' or 'yeast'
    
    Returns
    -------
    tuple of (bwa_ref_path, zap_ref_path, bwa_ref_dict, bwa_ref_lens, 
              zap_ref_dict, zap_ref_lens, ref_label_dict)
    """

    refs = files('trnazap').joinpath('references')
    
    if model == 'e_coli':
        bwa_ref = str(refs / 'bwa_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_bwa_subset.biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'eschColi_K_12_MG1655-mature-tRNAs_zap_ref.fa')
    elif model == 'yeast':
        bwa_ref = str(refs / 'bwa_align_references' / 'sacCer3-mature-tRNAs_bwa_subset_biosplints.fa')
        zap_ref = str(refs / 'zap_align_references' / 'sacCer3-mature-tRNAs_zap_ref.fa')
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'e_coli' or 'yeast'")
    
    # Load reference label dictionary
    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as infile:
        ref_label_dict = json.load(infile)

    if reference is None:
        # Load BWA references
        bwa_ref_dict = {}
        bwa_ref_lens = {}
        for seq in pysam.FastxFile(bwa_ref):
            label = ref_label_dict[seq.name]
            bwa_ref_dict[label] = seq.sequence
            bwa_ref_lens[label] = len(seq.sequence)
    else:
        # Load BWA references with custom reference path
        bwa_ref_dict = {}
        bwa_ref_lens = {}
        for seq in pysam.FastxFile(reference):
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
    
    return bwa_ref, zap_ref, bwa_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens, ref_label_dict


def print_alignment_summary(bwa_data, zap_data, comparison_data):
    """
    Print summary of alignment statistics.
    """
    print("\n" + "="*70)
    print("ALIGNMENT SUMMARY")
    print("="*70)
    print(f"BWA aligned reads:  {len(bwa_data['by_read']):>12,}")
    print(f"Zap aligned reads:  {len(zap_data['by_read']):>12,}")
    print(f"Both aligned:       {comparison_data['stats']['both_aligned']:>12,}")
    print(f"BWA only:           {comparison_data['stats']['bwa_only']:>12,}")
    print(f"Zap only:           {comparison_data['stats']['zap_only']:>12,}")
    print(f"\nAgreement rate:     {comparison_data['stats']['agreement_rate']:>12.2%}")
    print(f"Mean identity Δ:    {comparison_data['stats']['mean_identity_delta']:>+12.4f}")
    print("="*70 + "\n")


def generate_aligner_comparison_figures(
        reference,
        bwa_bam,
        zap_bam,
        model,
        out_dir,
        out_prefix='',
        threads=8,
        ident_threshold=0.75,
        min_coverage=25,
        # Plot flags
        per_class_identity=True,
        class_counts=True,
        class_count_deltas=True,
        per_position_error_heatmap=True,
        classification_heatmap=True,
        statistical_tests=True,
        length_identity_heatmaps=True,
        per_read_identity=True,
        misclassified_identity=True,
        one_aligner_only=True,
        summary_table=True,
        identity_histograms=True,
        positional_error_barplots=True
    ):
    """
    Complete aligner comparison pipeline: load data, compare, and generate all figures.
    
    Parameters
    ----------
    bwa_bam : str
        Path to BWA BAM file
    zap_bam : str
        Path to Zap BAM file
    model : str
        'e_coli' or 'yeast' - determines which package references to use
    out_dir : str
        Output directory for figures
    out_prefix : str
        Prefix for output files
    threads : int
        Number of threads for parallel processing
    ident_threshold : float
        Minimum identity threshold for filtering reads
    min_coverage : int
        Minimum alignment length for filtering reads
    per_class_identity : bool
        Generate per-class identity boxen plot
    class_counts : bool
        Generate class count bar plots (log and linear)
    class_count_deltas : bool
        Generate class count delta plot
    per_position_error_heatmap : bool
        Generate per-position error rate delta heatmap
    classification_heatmap : bool
        Generate alignment classification heatmap
    statistical_tests : bool
        Run statistical comparison tests (paired, unpaired, success rate)
    length_identity_heatmaps : bool
        Generate length vs identity heatmaps
    per_read_identity : bool
        Generate per-read identity 2D histogram
    misclassified_identity : bool
        Generate misclassified read identity 2D histogram
    one_aligner_only : bool
        Generate one-aligner-only identity histograms
    summary_table : bool
        Generate summary statistics table
    identity_histograms : bool
        Generate identity distribution histograms
    positional_error_barplots : bool
        Generate positional error stacked bar plots
    
    Returns
    -------
    dict containing:
        - 'bwa_data': BWA alignment data
        - 'zap_data': Zap alignment data
        - 'comparison_data': Comparison results
    """
    
    # Suppress plotting warnings
    suppress_plotting_warnings()
    
    # Suppress font warnings
    import warnings
    warnings.filterwarnings('ignore', message='1 extra bytes')
    warnings.filterwarnings('ignore', message='created.*timestamp')
    
    # Create output directory if needed
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"ALIGNER COMPARISON: {model.upper()}")
    print("="*70)
    
    # Load references from package
    print("\n" + "="*70)
    print("LOADING REFERENCE SEQUENCES")
    print("="*70)
    t0 = time()
    bwa_ref_path, zap_ref_path, bwa_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens, ref_label_dict = \
        load_references(model, reference)
    print(f"  BWA reference: {os.path.basename(bwa_ref_path)}")
    print(f"  Zap reference: {os.path.basename(zap_ref_path)}")
    print(f"  Loaded {len(bwa_ref_dict)} BWA references")
    print(f"  Loaded {len(zap_ref_dict)} Zap references")
    print(f"  Time: {(time() - t0):.2f} seconds")

    print(bwa_ref_dict)
    
    # Load BWA alignments
    print("\n" + "="*70)
    print("LOADING BWA ALIGNMENTS")
    print("="*70)
    t0 = time()
    bwa_data = load_alignments(
        bam_path=bwa_bam,
        ref_dict=bwa_ref_dict,
        ref_lens=bwa_ref_lens,
        five_offset=36,
        three_offset=42,
        model=model,
        threads=threads,
        ident_threshold=ident_threshold,
        min_coverage=min_coverage
    )
    print(f"  Aligned reads: {len(bwa_data['by_read']):,}")
    print(f"  Failed reads: {len(bwa_data['failed_reads']):,}")
    print(f"  Unmapped reads: {len(bwa_data['unmapped_reads']):,}")
    print(f"  Time: {(time() - t0) / 60:.2f} minutes")
    
    # Load Zap alignments
    print("\n" + "="*70)
    print("LOADING ZAP ALIGNMENTS")
    print("="*70)
    t0 = time()
    zap_data = load_alignments(
        bam_path=zap_bam,
        ref_dict=zap_ref_dict,
        ref_lens=zap_ref_lens,
        five_offset=0,
        three_offset=0,
        model=model,
        threads=threads,
        ident_threshold=ident_threshold,
        min_coverage=min_coverage
    )
    print(f"  Aligned reads: {len(zap_data['by_read']):,}")
    print(f"  Failed reads: {len(zap_data['failed_reads']):,}")
    print(f"  Unmapped reads: {len(zap_data['unmapped_reads']):,}")
    print(f"  Time: {(time() - t0) / 60:.2f} minutes")
    
    # Compare alignments
    print("\n" + "="*70)
    print("COMPARING ALIGNMENTS")
    print("="*70)
    t0 = time()
    comparison_data = compare_alignments_lightweight(bwa_data, zap_data)
    print(f"  Time: {(time() - t0):.2f} seconds")
    
    # Print summary
    print_alignment_summary(bwa_data, zap_data, comparison_data)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON FIGURES")
    print("="*70)
    
    if per_class_identity:
        print("  1. Per-class identity boxen plot...")
        plot_per_class_identity_boxen(bwa_data, zap_data, model, out_dir, out_prefix)
    
    if class_counts:
        print("  2. Class count bar plots (log + linear)...")
        plot_class_counts(bwa_data, zap_data, comparison_data, model, out_dir, out_prefix)
    
    if class_count_deltas:
        print("  3. Class count delta plot...")
        plot_class_count_deltas(bwa_data, zap_data, comparison_data, model, out_dir, out_prefix)
    
    if per_position_error_heatmap:
        print("  4. Per-position error rate delta heatmap...")
        plot_per_position_error_comparison_heatmap(bwa_data, zap_data, model, out_dir, out_prefix)
    
    if classification_heatmap:
        print("  5. Alignment classification heatmap...")
        plot_alignment_classification_heatmap(comparison_data, model, out_dir, out_prefix)
    
    if statistical_tests:
        print("  6. Statistical comparison tests...")
        run_statistical_comparisons(comparison_data, bwa_data, zap_data, out_dir, out_prefix)
    
    if length_identity_heatmaps:
        print("  7. Length vs identity heatmaps...")
        plot_length_identity_heatmaps(comparison_data, bwa_data, zap_data, out_dir, out_prefix)
    
    if per_read_identity:
        print("  8. Per-read identity 2D histogram...")
        plot_per_read_identity_2dhist(comparison_data, ident_threshold, out_dir, out_prefix)
    
    if misclassified_identity:
        print("  9. Misclassified read identity 2D histogram...")
        plot_misclassified_identity_2dhist(comparison_data, ident_threshold, out_dir, out_prefix)
    
    if one_aligner_only:
        print(" 10. One-aligner-only identity histograms...")
        plot_one_aligner_only_histograms(comparison_data, bwa_data, zap_data, 
                                        ident_threshold, out_dir, out_prefix)
    
    if summary_table:
        print(" 11. Summary statistics table...")
        create_summary_statistics_table(bwa_data, zap_data, model, out_dir, out_prefix)
    
    if identity_histograms:
        print(" 12. Identity histograms (3 versions)...")
        plot_identity_histograms(bwa_data, zap_data, out_dir=out_dir, out_prefix=out_prefix)
    
    if positional_error_barplots:
        print(" 13. Positional error stacked bar plots...")
        t0 = time()
        plot_positional_error_barplots(bwa_data, zap_data, model, out_dir, out_prefix)
        print(f"      Time: {(time() - t0) / 60:.2f} minutes")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"Output directory: {out_dir}")
    print(f"Output prefix: {out_prefix}")
    print("="*70 + "\n")
    
    return {
        'bwa_data': bwa_data,
        'zap_data': zap_data,
        'comparison_data': comparison_data
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python compare_aligners.py <bwa_bam> <zap_bam> <model> <out_dir> [out_prefix] [threads]")
        print("\nmodel: 'e_coli' or 'yeast'")
        sys.exit(1)
    
    bwa_bam = sys.argv[1]
    zap_bam = sys.argv[2]
    model = sys.argv[3]
    out_dir = sys.argv[4]
    out_prefix = sys.argv[5] if len(sys.argv) > 5 else ''
    threads = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    
    results = generate_aligner_comparison_figures(
        bwa_bam=bwa_bam,
        zap_bam=zap_bam,
        model=model,
        out_dir=out_dir,
        out_prefix=out_prefix,
        threads=threads
    )