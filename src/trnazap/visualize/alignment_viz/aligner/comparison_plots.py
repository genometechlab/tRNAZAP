"""
Plotting functions for aligner comparison.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
from importlib.resources import files

plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def suppress_plotting_warnings():
    """
    Suppress common matplotlib and numpy warnings during plotting.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')


def load_sort_order(model):
    """
    Load tRNA sort order from JSON file.
    """
    viz_path = files('trnazap').joinpath('visualize')
    
    if model == 'e_coli':
        json_path = viz_path / 'alignment_viz' / 'ecoli_label_sort_order.json'
    elif model == 'yeast':
        json_path = viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'
    else:
        json_path = viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'
    
    with open(str(json_path), 'r') as infile:
        sort_order = json.load(infile)
    
    return sort_order


def apply_sort_order(df, sort_order, column='tRNA'):
    """
    Apply custom sort order to DataFrame.
    """
    if column not in df.columns:
        return df
    
    df[column] = pd.Categorical(df[column], categories=sort_order, ordered=True)
    return df.sort_values(column)


# ============================================================================
# PLOT 1: Per-Class Identity Boxen Plot
# ============================================================================

def plot_per_class_identity_boxen(bwa_data, zap_data, model, 
                                  out_dir=None, out_prefix=''):
    """
    Per-class identity boxen plot comparing BWA vs Zap.
    """
    sort_order = load_sort_order(model)
    
    # Prepare data
    rows = []
    for trna in bwa_data['by_trna']:
        for ident in bwa_data['by_trna'][trna]['identities']:
            rows.append({'trna': trna, 'ident': ident, 'aligner': 'BWA'})
    
    for trna in zap_data['by_trna']:
        for ident in zap_data['by_trna'][trna]['identities']:
            rows.append({'trna': trna, 'ident': ident, 'aligner': 'Zap'})
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'trna')
    
    present_categories = [cat for cat in sort_order if cat in df['trna'].values]
    n_trnas = len(present_categories)
    fig_width = max(16, n_trnas * 0.3)
    
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.boxenplot(data=df, x='trna', y='ident', hue='aligner', ax=ax,
                 palette={'BWA': 'steelblue', 'Zap': 'orange'})
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Identity', fontsize=12)
    ax.set_title('Identity Distribution per tRNA Class', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.legend(title='Aligner')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}per_class_identity.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return df


# ============================================================================
# PLOT 2: Class Count Bar Plots (Log + Linear)
# ============================================================================

def plot_class_counts(bwa_data, zap_data, comparison_data, model, 
                     out_dir=None, out_prefix=''):
    """
    Side-by-side bar plots comparing read counts per tRNA class.
    Includes "Unmapped" category for reads aligned by only one aligner.
    Creates both log-scale and linear-scale versions.
    """
    sort_order = load_sort_order(model)
    
    # Get counts from by_trna (passing reads)
    rows = []
    for trna in bwa_data['by_trna']:
        rows.append({
            'trna': trna,
            'count': len(bwa_data['by_trna'][trna]['read_names']),
            'aligner': 'BWA'
        })
    
    for trna in zap_data['by_trna']:
        rows.append({
            'trna': trna,
            'count': len(zap_data['by_trna'][trna]['read_names']),
            'aligner': 'Zap'
        })
    
    # Add "Unmapped" category counts
    # BWA Unmapped = reads that Zap aligned but BWA didn't
    bwa_unmapped_count = len(comparison_data['read_sets']['zap_only'])
    rows.append({
        'trna': 'Unmapped',
        'count': bwa_unmapped_count,
        'aligner': 'BWA'
    })
    
    # Zap Unmapped = reads that BWA aligned but Zap didn't
    zap_unmapped_count = len(comparison_data['read_sets']['bwa_only'])
    rows.append({
        'trna': 'Unmapped',
        'count': zap_unmapped_count,
        'aligner': 'Zap'
    })
    
    count_df = pd.DataFrame(rows)
    
    # Sort: tRNAs in sort_order, then "Unmapped" at the end
    present_categories = [cat for cat in sort_order if cat in count_df['trna'].values]
    if 'Unmapped' in count_df['trna'].values and 'Unmapped' not in present_categories:
        present_categories.append('Unmapped')
    
    count_df['trna'] = pd.Categorical(count_df['trna'], 
                                      categories=present_categories, 
                                      ordered=True)
    count_df = count_df.sort_values('trna')
    
    n_trnas = len(present_categories)
    fig_width = max(12, n_trnas * 0.3)
    
    # Log-scale version
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.barplot(data=count_df, x='trna', y='count', hue='aligner', ax=ax,
               palette={'BWA': 'steelblue', 'Zap': 'orange'})
    ax.set_yscale('log')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Read Count (log scale)', fontsize=12)
    ax.set_title('Read Count Comparison (Log Scale)', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.legend(title='Aligner')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}class_counts_log.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Linear-scale version
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.barplot(data=count_df, x='trna', y='count', hue='aligner', ax=ax,
               palette={'BWA': 'steelblue', 'Zap': 'orange'})
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Read Count', fontsize=12)
    ax.set_title('Read Count Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.legend(title='Aligner')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}class_counts_linear.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return count_df


# ============================================================================
# PLOT 3: Class Count Delta Plot
# ============================================================================

def plot_class_count_deltas(bwa_data, zap_data, comparison_data, model, 
                           out_dir=None, out_prefix=''):
    """
    Bar plot showing difference in read counts (Zap - BWA) per tRNA class.
    Includes "Unmapped" category.
    """
    sort_order = load_sort_order(model)
    
    # Calculate deltas for tRNAs
    all_trnas = set(bwa_data['by_trna'].keys()) | set(zap_data['by_trna'].keys())
    
    rows = []
    for trna in all_trnas:
        bwa_count = len(bwa_data['by_trna'][trna]['read_names']) if trna in bwa_data['by_trna'] else 0
        zap_count = len(zap_data['by_trna'][trna]['read_names']) if trna in zap_data['by_trna'] else 0
        delta = zap_count - bwa_count
        
        rows.append({
            'trna': trna,
            'delta': delta
        })
    
    # Add "Unmapped" delta
    # Zap Unmapped count (reads BWA aligned but Zap didn't) - BWA Unmapped count (reads Zap aligned but BWA didn't)
    zap_unmapped = len(comparison_data['read_sets']['bwa_only'])
    bwa_unmapped = len(comparison_data['read_sets']['zap_only'])
    unmapped_delta = zap_unmapped - bwa_unmapped
    
    rows.append({
        'trna': 'Unmapped',
        'delta': unmapped_delta
    })
    
    df = pd.DataFrame(rows)
    
    # Sort: tRNAs in sort_order, then "Unmapped" at end
    present_trnas = [t for t in sort_order if t in df['trna'].values]
    if 'Unmapped' in df['trna'].values and 'Unmapped' not in present_trnas:
        present_trnas.append('Unmapped')
    
    df['trna'] = pd.Categorical(df['trna'], categories=present_trnas, ordered=True)
    df = df.sort_values('trna')
    
    n_trnas = len(df)
    fig_width = max(12, n_trnas * 0.3)
    
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ['red' if x < 0 else 'blue' for x in df['delta']]
    x_positions = np.arange(len(df))
    ax.bar(x_positions, df['delta'], color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['trna'], rotation=90, ha='right')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Δ Read Count (Zap - BWA)', fontsize=12)
    ax.set_title('Read Count Difference per tRNA Class', fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}class_count_deltas.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return df


# ============================================================================
# PLOT 4: Per-Position Error Rate Delta Heatmap
# ============================================================================

def plot_per_position_error_comparison_heatmap(bwa_data, zap_data, model,
                                               out_dir=None, out_prefix=''):
    """
    Single heatmap showing delta error rate (Zap - BWA) at each position for each tRNA.
    Positive values (red) = Zap has higher error rate
    Negative values (blue) = BWA has higher error rate
    """
    sort_order = load_sort_order(model)
    
    # Get tRNAs present in both
    all_trnas = set(bwa_data['by_trna'].keys()) & set(zap_data['by_trna'].keys())
    present_trnas = [trna for trna in sort_order if trna in all_trnas]
    
    if not present_trnas:
        print("No tRNAs found in both aligners for position error comparison")
        return None
    
    def calc_error_rates(track_arrs):
        """Calculate error rate at each position from stacked track_arrs"""
        # Sum across all reads - use nansum for matches!
        total_matches = np.nansum(track_arrs[:, 0, :], axis=0)
        total_coverage = np.sum(track_arrs[:, 2, :], axis=0)
        
        error_rate = np.zeros_like(total_coverage, dtype=float)
        mask = total_coverage > 0
        error_rate[mask] = 1 - (total_matches[mask] / total_coverage[mask])
        
        return error_rate
    
    # Find maximum length across all tRNAs
    max_len = max(
        max(bwa_data['by_trna'][t]['track_arrs'].shape[2] for t in present_trnas if t in bwa_data['by_trna']),
        max(zap_data['by_trna'][t]['track_arrs'].shape[2] for t in present_trnas if t in zap_data['by_trna'])
    )
    
    # Build delta matrix
    delta_matrix = np.full((len(present_trnas), max_len), np.nan)
    
    for idx, trna in enumerate(present_trnas):
        if trna in bwa_data['by_trna'] and trna in zap_data['by_trna']:
            bwa_error = calc_error_rates(bwa_data['by_trna'][trna]['track_arrs'])
            zap_error = calc_error_rates(zap_data['by_trna'][trna]['track_arrs'])
            
            # Calculate delta (Zap - BWA)
            min_len = min(len(bwa_error), len(zap_error))
            delta_matrix[idx, :min_len] = zap_error[:min_len] - bwa_error[:min_len]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, max(10, len(present_trnas) * 0.3)))
    
    # Use diverging colormap: blue = BWA worse, red = Zap worse
    # Set symmetric limits
    vmax = np.nanmax(np.abs(delta_matrix))
    vmax = min(vmax, 0.2)  # Cap at 20% difference for better visualization
    
    sns.heatmap(delta_matrix, cmap='RdBu_r', center=0, 
                vmin=-vmax, vmax=vmax, ax=ax,
                yticklabels=present_trnas, 
                cbar_kws={'label': 'Δ Error Rate (Zap - BWA)'})
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('tRNA', fontsize=12)
    ax.set_title('Per-Position Error Rate Comparison (Zap - BWA)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotation explaining the colors
    ax.text(0.02, 0.98, 'Blue = BWA higher error | Red = Zap higher error',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}per_position_error_delta.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Save TSV of delta values
        df = pd.DataFrame(delta_matrix, index=present_trnas)
        tsv_path = os.path.join(out_dir, f'{out_prefix}per_position_error_delta.tsv')
        df.to_csv(tsv_path, sep='\t')
        print(f"Saved delta error rate data: {tsv_path}")
    
    plt.close(fig)
    
    return {'delta_matrix': delta_matrix, 'trnas': present_trnas}


# ============================================================================
# PLOT 5: Alignment Classification Heatmap
# ============================================================================

def plot_alignment_classification_heatmap(comparison_data, model,
                                         out_dir=None, out_prefix=''):
    """
    Heatmap showing how BWA classifications correspond to Zap classifications.
    Now includes "Unmapped" for reads aligned by only one aligner.
    """
    sort_order = load_sort_order(model)
    
    # Build count matrix - now includes ALL reads in read_comparison
    bwa_classes = []
    zap_classes = []
    
    for read_id, info in comparison_data['read_comparison'].items():
        bwa_classes.append(info['bwa_trna'])
        zap_classes.append(info['zap_trna'])
    
    # Create crosstab
    df = pd.DataFrame({'bwa': bwa_classes, 'zap': zap_classes})
    
    # Add "Unmapped" to the categories
    all_classes = set(bwa_classes) | set(zap_classes)
    df['bwa'] = pd.Categorical(df['bwa'], categories=all_classes)
    df['zap'] = pd.Categorical(df['zap'], categories=all_classes)
    
    count_matrix = pd.crosstab(df['bwa'], df['zap'], 
                               rownames=['BWA'], colnames=['Zap'],
                               dropna=False)
    
    # Zero out the Unmapped-Unmapped cell (we don't track reads both failed)
    if 'Unmapped' in count_matrix.index and 'Unmapped' in count_matrix.columns:
        count_matrix.loc['Unmapped', 'Unmapped'] = 0
    
    # Sort with "Unmapped" at the end
    present_rows = [cat for cat in sort_order if cat in count_matrix.index]
    present_cols = [cat for cat in sort_order if cat in count_matrix.columns]
    
    # Add "Unmapped" to the end if present
    if 'Unmapped' in count_matrix.index and 'Unmapped' not in present_rows:
        present_rows.append('Unmapped')
    if 'Unmapped' in count_matrix.columns and 'Unmapped' not in present_cols:
        present_cols.append('Unmapped')
    
    count_matrix = count_matrix.reindex(index=present_rows, columns=present_cols, fill_value=0)

    # Save count matrix
    matrix_path = os.path.join(out_dir, f'{out_prefix}_aligner_comparison_count_matrix.tsv')
    count_matrix.to_csv(matrix_path, sep='\t')
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(14, len(present_cols) * 0.3),
                                   max(12, len(present_rows) * 0.3)))
    
    sns.heatmap(count_matrix, norm=LogNorm(), cmap='Blues', 
               cbar=True, square=True, ax=ax,
               xticklabels=present_cols, yticklabels=present_rows,
               linewidths=0, linecolor='none')
    
    ax.set_facecolor('white')
    ax.set_xlabel('Zap Classification', fontsize=12)
    ax.set_ylabel('BWA Classification', fontsize=12)
    ax.set_title('Alignment Classification Comparison', fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}alignment_classification_heatmap.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return count_matrix


# ============================================================================
# PLOT 6: Statistical Comparison Tests
# ============================================================================

def run_statistical_comparisons(comparison_data, bwa_data, zap_data,
                                out_dir=None, out_prefix=''):
    """
    Run complete statistical comparison suite:
    - Paired tests (Wilcoxon) for reads aligned by both
    - Unpaired tests (Mann-Whitney U) for all reads
    - Success rate comparison
    """
    
    def paired_identity_test(bwa_idents, zap_idents, margin=0.00, test_name=""):
        """Paired test using Wilcoxon signed-rank."""
        differences = zap_idents - bwa_idents
        w_stat, p_value = stats.wilcoxon(differences - margin, alternative='greater')
        
        mean_bwa = np.mean(bwa_idents)
        mean_zap = np.mean(zap_idents)
        mean_diff = np.mean(differences)
        
        median_bwa = np.median(bwa_idents)
        median_zap = np.median(zap_idents)
        median_diff = np.median(differences)
        
        significant = p_value < 0.05
        
        if test_name:
            print(f"\n{test_name}")
        print(f"  N={len(differences):,}")
        print(f"  BWA: mean={mean_bwa:.4f}, median={median_bwa:.4f}")
        print(f"  Zap: mean={mean_zap:.4f}, median={median_zap:.4f}")
        print(f"  Δ: mean={mean_diff:+.4f}, median={median_diff:+.4f}")
        print(f"  p={p_value:.4e} | {'✓ Zap superior' if significant else '✗ Not superior'}")
        
        return {
            'test_type': 'paired_wilcoxon',
            'test_name': test_name,
            'n': len(differences),
            'mean_bwa': mean_bwa,
            'mean_zap': mean_zap,
            'mean_diff': mean_diff,
            'median_bwa': median_bwa,
            'median_zap': median_zap,
            'median_diff': median_diff,
            'p_value': p_value,
            'significant': significant,
            'w_statistic': w_stat,
            'margin': margin
        }
    
    def unpaired_identity_test(bwa_idents, zap_idents, test_name=""):
        """Unpaired test using Mann-Whitney U."""
        u_stat, p_value = stats.mannwhitneyu(zap_idents, bwa_idents, 
                                              alternative='greater')
        
        mean_bwa = np.mean(bwa_idents)
        mean_zap = np.mean(zap_idents)
        mean_diff = mean_zap - mean_bwa
        
        median_bwa = np.median(bwa_idents)
        median_zap = np.median(zap_idents)
        median_diff = median_zap - median_bwa
        
        significant = p_value < 0.05
        
        if test_name:
            print(f"\n{test_name}")
        print(f"  N_BWA={len(bwa_idents):,} | N_Zap={len(zap_idents):,}")
        print(f"  BWA: mean={mean_bwa:.4f}, median={median_bwa:.4f}")
        print(f"  Zap: mean={mean_zap:.4f}, median={median_zap:.4f}")
        print(f"  Δ: mean={mean_diff:+.4f}, median={median_diff:+.4f}")
        print(f"  p={p_value:.4e} | {'✓ Zap superior' if significant else '✗ Not superior'}")
        
        return {
            'test_type': 'unpaired_mannwhitney',
            'test_name': test_name,
            'n_bwa': len(bwa_idents),
            'n_zap': len(zap_idents),
            'mean_bwa': mean_bwa,
            'mean_zap': mean_zap,
            'mean_diff': mean_diff,
            'median_bwa': median_bwa,
            'median_zap': median_zap,
            'median_diff': median_diff,
            'p_value': p_value,
            'significant': significant,
            'u_statistic': u_stat
        }
    
    print("\n" + "="*70)
    print("STATISTICAL COMPARISONS: Zap vs BWA")
    print("="*70)
    
    # ========================================================================
    # PAIRED TESTS (Wilcoxon) - Only reads aligned by BOTH
    # ========================================================================
    print("\n" + "-"*70)
    print("PAIRED TESTS (Wilcoxon Signed-Rank)")
    print("-"*70)
    
    # Test 1a: All reads aligned by both
    bwa_idents_both = []
    zap_idents_both = []
    for read_id, info in comparison_data['read_comparison'].items():
        if info['bwa_identity'] is not None and info['zap_identity'] is not None:
            bwa_idents_both.append(info['bwa_identity'])
            zap_idents_both.append(info['zap_identity'])
    
    result_paired_all = paired_identity_test(
        np.array(bwa_idents_both),
        np.array(zap_idents_both),
        margin=0.00,
        test_name="All reads aligned by BOTH aligners"
    )
    
    # Test 1b: Reads where aligners agree on classification
    bwa_idents_agree = []
    zap_idents_agree = []
    for read_id in comparison_data['read_sets']['agree']:
        info = comparison_data['read_comparison'][read_id]
        bwa_idents_agree.append(info['bwa_identity'])
        zap_idents_agree.append(info['zap_identity'])
    
    result_paired_agree = paired_identity_test(
        np.array(bwa_idents_agree),
        np.array(zap_idents_agree),
        margin=0.00,
        test_name="Reads where aligners AGREE on tRNA classification"
    )
    
    # Test 1c: Reads where aligners disagree on classification
    bwa_idents_disagree = []
    zap_idents_disagree = []
    for read_id in comparison_data['read_sets']['disagree']:
        info = comparison_data['read_comparison'][read_id]
        bwa_idents_disagree.append(info['bwa_identity'])
        zap_idents_disagree.append(info['zap_identity'])
    
    result_paired_disagree = paired_identity_test(
        np.array(bwa_idents_disagree),
        np.array(zap_idents_disagree),
        margin=0.00,
        test_name="Reads where aligners DISAGREE on tRNA classification"
    )
    
    # ========================================================================
    # UNPAIRED TESTS (Mann-Whitney U) - ALL reads each aligner aligned
    # ========================================================================
    print("\n" + "-"*70)
    print("UNPAIRED TESTS (Mann-Whitney U)")
    print("-"*70)
    
    # Collect ALL identities from each aligner
    bwa_all_idents = []
    for read_id, info in bwa_data['by_read'].items():
        bwa_all_idents.append(info['identity'])
    
    zap_all_idents = []
    for read_id, info in zap_data['by_read'].items():
        zap_all_idents.append(info['identity'])
    
    result_unpaired_all = unpaired_identity_test(
        np.array(bwa_all_idents),
        np.array(zap_all_idents),
        test_name="ALL reads (including one-aligner-only)"
    )
    
    # ========================================================================
    # SUCCESS RATE COMPARISON
    # ========================================================================
    print("\n" + "-"*70)
    print("ALIGNMENT SUCCESS RATE")
    print("-"*70)
    
    total_bwa = len(bwa_data['by_read'])
    total_zap = len(zap_data['by_read'])
    
    # Total reads attempted - union of all read sets
    all_read_ids = set()
    all_read_ids.update(bwa_data['by_read'].keys())
    all_read_ids.update(bwa_data['failed_reads'])
    all_read_ids.update(bwa_data['unmapped_reads'])
    all_read_ids.update(zap_data['by_read'].keys())
    all_read_ids.update(zap_data['failed_reads'])
    all_read_ids.update(zap_data['unmapped_reads'])
    
    total_reads = len(all_read_ids)
    
    bwa_success_rate = total_bwa / total_reads
    zap_success_rate = total_zap / total_reads
    
    # Chi-square test for success rates
    observed = np.array([[total_bwa, len(bwa_data['failed_reads']) + len(bwa_data['unmapped_reads'])],
                        [total_zap, len(zap_data['failed_reads']) + len(zap_data['unmapped_reads'])]])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    significant = p_value < 0.05
    
    print(f"\nAlignment Success Rates")
    print(f"  Total reads processed: {total_reads:,}")
    print(f"  BWA: {total_bwa:,} / {total_reads:,} = {bwa_success_rate:.2%}")
    print(f"  Zap: {total_zap:,} / {total_reads:,} = {zap_success_rate:.2%}")
    print(f"  Δ: {(zap_success_rate - bwa_success_rate)*100:+.2f} percentage points")
    print(f"  χ² = {chi2:.2f}, p = {p_value:.4e} | {'✓ Significant' if significant else '✗ Not significant'}")
    
    result_success_rate = {
        'test_type': 'success_rate_chi2',
        'test_name': 'Alignment success rate comparison',
        'total_reads': total_reads,
        'bwa_success': total_bwa,
        'zap_success': total_zap,
        'bwa_success_rate': bwa_success_rate,
        'zap_success_rate': zap_success_rate,
        'success_rate_diff': zap_success_rate - bwa_success_rate,
        'chi2': chi2,
        'p_value': p_value,
        'significant': significant
    }
    
    print("="*70 + "\n")
    
    # ========================================================================
    # Compile all results
    # ========================================================================
    results = {
        'paired_all_reads': result_paired_all,
        'paired_agree_reads': result_paired_agree,
        'paired_disagree_reads': result_paired_disagree,
        'unpaired_all_reads': result_unpaired_all,
        'success_rate': result_success_rate
    }
    
    # Save to CSV
    if out_dir:
        rows = []
        
        # Paired tests
        for key in ['paired_all_reads', 'paired_agree_reads', 'paired_disagree_reads']:
            result = results[key]
            rows.append({
                'Test_Type': result['test_type'],
                'Test': result['test_name'],
                'N_reads': result['n'],
                'BWA_mean_identity': result['mean_bwa'],
                'Zap_mean_identity': result['mean_zap'],
                'Mean_diff': result['mean_diff'],
                'BWA_median_identity': result['median_bwa'],
                'Zap_median_identity': result['median_zap'],
                'Median_diff': result['median_diff'],
                'P_value': result['p_value'],
                'Significant': result['significant'],
                'Statistic': result['w_statistic'],
                'Margin': result['margin']
            })
        
        # Unpaired test
        result = results['unpaired_all_reads']
        rows.append({
            'Test_Type': result['test_type'],
            'Test': result['test_name'],
            'N_reads': f"BWA:{result['n_bwa']}, Zap:{result['n_zap']}",
            'BWA_mean_identity': result['mean_bwa'],
            'Zap_mean_identity': result['mean_zap'],
            'Mean_diff': result['mean_diff'],
            'BWA_median_identity': result['median_bwa'],
            'Zap_median_identity': result['median_zap'],
            'Median_diff': result['median_diff'],
            'P_value': result['p_value'],
            'Significant': result['significant'],
            'Statistic': result['u_statistic'],
            'Margin': 'N/A'
        })
        
        # Success rate test
        result = results['success_rate']
        rows.append({
            'Test_Type': result['test_type'],
            'Test': result['test_name'],
            'N_reads': result['total_reads'],
            'BWA_mean_identity': 'N/A',
            'Zap_mean_identity': 'N/A',
            'Mean_diff': result['success_rate_diff'],
            'BWA_median_identity': 'N/A',
            'Zap_median_identity': 'N/A',
            'Median_diff': 'N/A',
            'P_value': result['p_value'],
            'Significant': result['significant'],
            'Statistic': result['chi2'],
            'Margin': 'N/A'
        })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{out_prefix}_statistical_tests.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved statistical test results: {csv_path}")
    
    return results


# ============================================================================
# PLOT 7: Alignment Length vs Identity Heatmaps
# ============================================================================

def plot_length_identity_heatmaps(comparison_data, bwa_data, zap_data,
                                 out_dir=None, out_prefix='',
                                 bins_length=50, bins_identity=50):
    """
    2D histogram heatmaps showing alignment length vs identity.
    Creates three plots: BWA, Zap, and Delta.
    """
    # Extract data for all reads aligned by each aligner
    bwa_lengths = []
    bwa_idents = []
    for read_id, info in bwa_data['by_read'].items():
        bwa_lengths.append(info['length'])
        bwa_idents.append(info['identity'])
    
    zap_lengths = []
    zap_idents = []
    for read_id, info in zap_data['by_read'].items():
        zap_lengths.append(info['length'])
        zap_idents.append(info['identity'])
    
    bwa_lengths = np.array(bwa_lengths)
    bwa_idents = np.array(bwa_idents)
    zap_lengths = np.array(zap_lengths)
    zap_idents = np.array(zap_idents)
    
    # Define consistent bins for all plots
    all_lengths = np.concatenate([bwa_lengths, zap_lengths])
    all_idents = np.concatenate([bwa_idents, zap_idents])
    
    length_bins = np.linspace(all_lengths.min(), all_lengths.max(), bins_length + 1)
    identity_bins = np.linspace(all_idents.min(), all_idents.max(), bins_identity + 1)
    
    # Create 2D histograms
    bwa_hist, _, _ = np.histogram2d(bwa_lengths, bwa_idents, 
                                     bins=[length_bins, identity_bins])
    zap_hist, _, _ = np.histogram2d(zap_lengths, zap_idents,
                                     bins=[length_bins, identity_bins])
    delta_hist = zap_hist - bwa_hist
    
    # Plot BWA
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(bwa_hist.T, origin='lower', aspect='auto', cmap='Blues',
                   extent=[length_bins[0], length_bins[-1], 
                          identity_bins[0], identity_bins[-1]],
                   norm=LogNorm(vmin=max(1, bwa_hist[bwa_hist > 0].min()), 
                               vmax=bwa_hist.max()))
    ax.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax.set_ylabel('Identity', fontsize=12)
    ax.set_title('BWA Alignments: Length vs Identity', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count (log scale)')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}length_identity_heatmap_bwa.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot Zap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(zap_hist.T, origin='lower', aspect='auto', cmap='Oranges',
                   extent=[length_bins[0], length_bins[-1],
                          identity_bins[0], identity_bins[-1]],
                   norm=LogNorm(vmin=max(1, zap_hist[zap_hist > 0].min()), 
                               vmax=zap_hist.max()))
    ax.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax.set_ylabel('Identity', fontsize=12)
    ax.set_title('tRNA-zap Alignments: Length vs Identity', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count (log scale)')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}length_identity_heatmap_zap.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot Delta
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.abs(delta_hist).max()
    im = ax.imshow(delta_hist.T, origin='lower', aspect='auto', cmap='RdBu_r',
                   extent=[length_bins[0], length_bins[-1],
                          identity_bins[0], identity_bins[-1]],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax.set_ylabel('Identity', fontsize=12)
    ax.set_title('Delta (Zap - BWA): Length vs Identity', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count Difference')
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}length_identity_heatmap_delta.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'bwa_hist': bwa_hist,
        'zap_hist': zap_hist,
        'delta_hist': delta_hist,
        'length_bins': length_bins,
        'identity_bins': identity_bins
    }


# ============================================================================
# PLOT 8: Per-Read Identity 2D Histogram
# ============================================================================

def rmse_numpy(actual, predicted):
    """
    Calculates the Root Mean Square Error (RMSE) between two numpy vectors.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    # Calculate the mean of the squared differences
    mse = np.mean((actual - predicted)**2) 
    # Take the square root
    rmse = np.sqrt(mse)
    return rmse

def plot_per_read_identity_2dhist(comparison_data, ident_threshold=0.75,
                                  out_dir=None, out_prefix=''):
    """
    2D histogram comparing BWA vs Zap identity for reads aligned by both.
    Uses 1% bins starting from floored ident_threshold.
    Zap on x-axis, BWA on y-axis.
    """
    # Get identities for reads aligned by BOTH (skip Unmapped)
    bwa_idents = []
    zap_idents = []
    
    for read_id, info in comparison_data['read_comparison'].items():
        # Skip reads where either aligner has Unmapped
        if info['bwa_identity'] is not None and info['zap_identity'] is not None:
            bwa_idents.append(info['bwa_identity'])
            zap_idents.append(info['zap_identity'])
    
    df = pd.DataFrame({
        'bwa_ident': bwa_idents,
        'zap_ident': zap_idents
    })

    print(f"Per Read Alignment Identity RMSE: {rmse_numpy(np.array(df['bwa_ident'].to_list()), np.array(df['zap_ident'].to_list()))}")
    
    # Create 1% bins starting from floored ident_threshold
    bin_start = np.floor(ident_threshold * 100) / 100  # Round down to nearest 0.01
    bins = np.arange(bin_start, 1.0 + 0.01, 0.01)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use histogram2d with explicit bins
    # Zap on x-axis (first arg), BWA on y-axis (second arg)
    h = ax.hist2d(df['zap_ident'], df['bwa_ident'],
                  bins=[bins, bins], cmap='viridis', cmin=1)
    
    # Add colorbar
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Add diagonal line
    ax.plot([bin_start, 1], [bin_start, 1], 
            'r--', alpha=0.5, linewidth=2, label='x=y')
    
    ax.set_xlabel('Zap Identity', fontsize=12)
    ax.set_ylabel('BWA Identity', fontsize=12)
    ax.set_title(f'Per-Read Identity Comparison (N={len(df):,})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([bin_start, 1.0])
    ax.set_ylim([bin_start, 1.0])
    ax.set_aspect('equal', adjustable='box')  # Make plot square
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}per_read_identity.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return df


# ============================================================================
# PLOT 9: Misclassified Read Identity 2D Histogram
# ============================================================================

def plot_misclassified_identity_2dhist(comparison_data, ident_threshold=0.75,
                                       out_dir=None, out_prefix=''):
    """
    2D histogram for reads where aligners disagree on tRNA classification.
    Uses 1% bins starting from floored ident_threshold.
    Zap on x-axis, BWA on y-axis.
    """
    # Get identities for disagreed reads
    bwa_idents = []
    zap_idents = []
    
    for read_id in comparison_data['read_sets']['disagree']:
        info = comparison_data['read_comparison'][read_id]
        bwa_idents.append(info['bwa_identity'])
        zap_idents.append(info['zap_identity'])
    
    if len(bwa_idents) == 0:
        print("No misclassified reads found")
        return None
    
    df = pd.DataFrame({
        'bwa_ident': bwa_idents,
        'zap_ident': zap_idents
    })
    
    # Create 1% bins starting from floored ident_threshold
    bin_start = np.floor(ident_threshold * 100) / 100  # Round down to nearest 0.01
    bins = np.arange(bin_start, 1.0 + 0.01, 0.01)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use histogram2d with explicit bins
    # Zap on x-axis (first arg), BWA on y-axis (second arg)
    h = ax.hist2d(df['zap_ident'], df['bwa_ident'], 
                  bins=[bins, bins], cmap='Reds', cmin=1)
    
    # Add colorbar
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Add diagonal line
    ax.plot([bin_start, 1], [bin_start, 1], 
            'k--', alpha=0.5, linewidth=2, label='x=y')
    
    ax.set_xlabel('Zap Identity', fontsize=12)
    ax.set_ylabel('BWA Identity', fontsize=12)
    ax.set_title(f'Misclassified Reads Identity (N={len(df):,})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([bin_start, 1.0])
    ax.set_ylim([bin_start, 1.0])
    ax.set_aspect('equal', adjustable='box')  # Make plot square
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}misclassified_read_identity.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return df


# ============================================================================
# PLOT 10: One-Aligner-Only Identity Histograms
# ============================================================================

def plot_one_aligner_only_histograms(comparison_data, bwa_data, zap_data,
                                     ident_threshold=0.75,
                                     out_dir=None, out_prefix=''):
    """
    Overlaid histograms of identity for reads aligned by only one aligner.
    Uses 1% bins starting from floored ident_threshold.
    """
    # Get identities for BWA-only reads
    bwa_only_idents = []
    for read_id in comparison_data['read_sets']['bwa_only']:
        if read_id in bwa_data['by_read']:
            bwa_only_idents.append(bwa_data['by_read'][read_id]['identity'])
    
    # Get identities for Zap-only reads
    zap_only_idents = []
    for read_id in comparison_data['read_sets']['zap_only']:
        if read_id in zap_data['by_read']:
            zap_only_idents.append(zap_data['by_read'][read_id]['identity'])
    
    if len(bwa_only_idents) == 0 and len(zap_only_idents) == 0:
        print("No reads aligned by only one aligner")
        return None
    
    # Create 1% bins starting from floored ident_threshold
    bin_start = np.floor(ident_threshold * 100) / 100  # Round down to nearest 0.01
    bins = np.arange(bin_start, 1.0 + 0.01, 0.01)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(bwa_only_idents) > 0:
        ax.hist(bwa_only_idents, bins=bins, 
               label=f'BWA only (N={len(bwa_only_idents):,})', 
               color='steelblue', alpha=0.6)
    
    if len(zap_only_idents) > 0:
        ax.hist(zap_only_idents, bins=bins,
               label=f'Zap only (N={len(zap_only_idents):,})', 
               color='orange', alpha=0.6)
    
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Identity Distribution: Reads Aligned by Only One Aligner', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([bin_start, 1.0])
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}one_aligner_only_identity.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'bwa_only_idents': bwa_only_idents,
        'zap_only_idents': zap_only_idents
    }


# ============================================================================
# PLOT 11: Summary Statistics Table
# ============================================================================

def create_summary_statistics_table(bwa_data, zap_data, model,
                                   out_dir=None, out_prefix=''):
    """
    Create comprehensive summary statistics table per tRNA class.
    """
    sort_order = load_sort_order(model)
    
    def calculate_error_proportions(track_arrs):
        """Calculate error proportions from stacked track arrays."""
        matches = np.nansum(track_arrs[:, 0, :])
        insertions = np.nansum(track_arrs[:, 1, :])
        coverage = np.nansum(track_arrs[:, 2, :])
        deletions = np.nansum(track_arrs[:, 3, :])
        
        # Mismatches = covered positions that aren't matches or deletions
        mismatches = coverage - matches - deletions
        
        total = matches + mismatches + insertions + deletions
        
        if total > 0:
            return {
                'matches': matches,
                'mismatches': mismatches,
                'insertions': insertions,
                'deletions': deletions,
                'prop_match': matches / total,
                'prop_mismatch': mismatches / total,
                'prop_insertion': insertions / total,
                'prop_deletion': deletions / total
            }
        else:
            return {
                'matches': 0, 'mismatches': 0, 'insertions': 0, 'deletions': 0,
                'prop_match': 0, 'prop_mismatch': 0, 'prop_insertion': 0, 'prop_deletion': 0
            }
    
    # Get all tRNA classes
    all_trnas = set(bwa_data['by_trna'].keys()) | set(zap_data['by_trna'].keys())
    all_trnas = [t for t in sort_order if t in all_trnas]
    
    summary_data = []
    
    # Calculate stats for each tRNA class
    for trna in all_trnas:
        # BWA stats
        if trna in bwa_data['by_trna']:
            bwa_idents = bwa_data['by_trna'][trna]['identities']
            bwa_lengths = bwa_data['by_trna'][trna]['alignment_lengths']
            bwa_track_arrs = bwa_data['by_trna'][trna]['track_arrs']
            
            error_props = calculate_error_proportions(bwa_track_arrs)
            ref_len = bwa_track_arrs.shape[2]
            
            summary_data.append({
                'tRNA': trna,
                'Aligner': 'BWA',
                'N_reads': len(bwa_idents),
                'Identity_mean': np.mean(bwa_idents),
                'Identity_std': np.std(bwa_idents),
                'Identity_25th': np.percentile(bwa_idents, 25),
                'Identity_median': np.median(bwa_idents),
                'Identity_75th': np.percentile(bwa_idents, 75),
                'AlignLen_mean': np.mean(bwa_lengths),
                'AlignLen_median': np.median(bwa_lengths),
                'RefLen': ref_len,
                'Prop_match': error_props['prop_match'],
                'Prop_mismatch': error_props['prop_mismatch'],
                'Prop_insertion': error_props['prop_insertion'],
                'Prop_deletion': error_props['prop_deletion']
            })
        
        # Zap stats
        if trna in zap_data['by_trna']:
            zap_idents = zap_data['by_trna'][trna]['identities']
            zap_lengths = zap_data['by_trna'][trna]['alignment_lengths']
            zap_track_arrs = zap_data['by_trna'][trna]['track_arrs']
            
            error_props = calculate_error_proportions(zap_track_arrs)
            ref_len = zap_track_arrs.shape[2]
            
            summary_data.append({
                'tRNA': trna,
                'Aligner': 'Zap',
                'N_reads': len(zap_idents),
                'Identity_mean': np.mean(zap_idents),
                'Identity_std': np.std(zap_idents),
                'Identity_25th': np.percentile(zap_idents, 25),
                'Identity_median': np.median(zap_idents),
                'Identity_75th': np.percentile(zap_idents, 75),
                'AlignLen_mean': np.mean(zap_lengths),
                'AlignLen_median': np.median(zap_lengths),
                'RefLen': ref_len,
                'Prop_match': error_props['prop_match'],
                'Prop_mismatch': error_props['prop_mismatch'],
                'Prop_insertion': error_props['prop_insertion'],
                'Prop_deletion': error_props['prop_deletion']
            })
    
    df = pd.DataFrame(summary_data)
    
    # Add "All tRNAs" summary row for each aligner
    for aligner_name, data in [('BWA', bwa_data), ('Zap', zap_data)]:
        aligner_df = df[df['Aligner'] == aligner_name]
        
        if len(aligner_df) > 0:
            # Aggregate all reads
            all_idents = []
            all_lengths = []
            
            # Aggregate error counts across all tRNAs (don't concatenate arrays)
            total_matches = 0
            total_mismatches = 0
            total_insertions = 0
            total_deletions = 0
            
            for trna in all_trnas:
                if trna in data['by_trna']:
                    all_idents.extend(data['by_trna'][trna]['identities'])
                    all_lengths.extend(data['by_trna'][trna]['alignment_lengths'])
                    
                    # Calculate error counts for this tRNA
                    track_arrs = data['by_trna'][trna]['track_arrs']
                    matches = np.nansum(track_arrs[:, 0, :])
                    insertions = np.nansum(track_arrs[:, 1, :])
                    coverage = np.nansum(track_arrs[:, 2, :])
                    deletions = np.nansum(track_arrs[:, 3, :])
                    mismatches = coverage - matches - deletions
                    
                    total_matches += matches
                    total_mismatches += mismatches
                    total_insertions += insertions
                    total_deletions += deletions
            
            all_idents = np.array(all_idents)
            all_lengths = np.array(all_lengths)
            
            # Calculate proportions
            total_events = total_matches + total_mismatches + total_insertions + total_deletions
            if total_events > 0:
                prop_match = total_matches / total_events
                prop_mismatch = total_mismatches / total_events
                prop_insertion = total_insertions / total_events
                prop_deletion = total_deletions / total_events
            else:
                prop_match = prop_mismatch = prop_insertion = prop_deletion = 0
            
            summary_row = {
                'tRNA': 'All_tRNAs',
                'Aligner': aligner_name,
                'N_reads': len(all_idents),
                'Identity_mean': np.mean(all_idents),
                'Identity_std': np.std(all_idents),
                'Identity_25th': np.percentile(all_idents, 25),
                'Identity_median': np.median(all_idents),
                'Identity_75th': np.percentile(all_idents, 75),
                'AlignLen_mean': np.mean(all_lengths),
                'AlignLen_median': np.median(all_lengths),
                'RefLen': np.mean(aligner_df['RefLen']),
                'Prop_match': prop_match,
                'Prop_mismatch': prop_mismatch,
                'Prop_insertion': prop_insertion,
                'Prop_deletion': prop_deletion
            }
            
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Sort
    df['sort_key'] = df['tRNA'].apply(
        lambda x: (1, 0) if x == 'All_tRNAs' else (0, sort_order.index(x) if x in sort_order else 999)
    )
    df = df.sort_values(['sort_key', 'Aligner']).drop('sort_key', axis=1)
    
    # Save
    if out_dir:
        csv_path = os.path.join(out_dir, f"{out_prefix}_summary_statistics.csv")
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved summary statistics table: {csv_path}")
    
    return df


# ============================================================================
# PLOT 12: Identity Histograms (3 versions)
# ============================================================================

def plot_identity_histograms(bwa_data, zap_data, binwidth=0.01,
                             out_dir=None, out_prefix=''):
    """
    Create identity histograms: BWA only, Zap only, and overlaid.
    Uses 1% bins.
    """
    # Collect all identities
    bwa_all_idents = []
    for trna in bwa_data['by_trna']:
        bwa_all_idents.extend(bwa_data['by_trna'][trna]['identities'])
    
    zap_all_idents = []
    for trna in zap_data['by_trna']:
        zap_all_idents.extend(zap_data['by_trna'][trna]['identities'])
    
    # BWA only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bwa_all_idents, bins=np.arange(0, 1.0 + binwidth, binwidth),
           color='steelblue', alpha=0.75, label='BWA')
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('BWA Alignment Identity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}identity_histogram_bwa.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Zap only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(zap_all_idents, bins=np.arange(0, 1.0 + binwidth, binwidth),
           color='orange', alpha=0.75, label='Zap')
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('tRNA-zap Alignment Identity Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}identity_histogram_zap.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bwa_all_idents, bins=np.arange(0, 1.0 + binwidth, binwidth),
           color='steelblue', alpha=0.5, label='BWA')
    ax.hist(zap_all_idents, bins=np.arange(0, 1.0 + binwidth, binwidth),
           color='orange', alpha=0.5, label='Zap')
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Alignment Identity Distribution Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}identity_histogram_overlay.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'bwa_idents': bwa_all_idents,
        'zap_idents': zap_all_idents
    }


# ============================================================================
# PLOT 13: Positional Error Stacked Bar Plots
# ============================================================================

def plot_positional_error_barplots(bwa_data, zap_data, model,
                                   out_dir=None, out_prefix=''):
    """
    Stacked bar plots showing match/mismatch/insertion/deletion proportions
    at each position for each tRNA.
    """
    sort_order = load_sort_order(model)
    
    # Get tRNAs present in both
    all_trnas = set(bwa_data['by_trna'].keys()) & set(zap_data['by_trna'].keys())
    present_trnas = [trna for trna in sort_order if trna in all_trnas]
    
    if not present_trnas:
        print("No tRNAs found in both aligners for positional error plots")
        return None
    
    def compute_position_error_props(track_arrs):
        """Compute error proportions at each position."""
        # Sum across all reads
        # CRITICAL: Use nansum for matches because row 0 contains NaN for uncovered positions
        matches = np.nansum(track_arrs[:, 0, :], axis=0)
        insertions = np.sum(track_arrs[:, 1, :], axis=0)
        coverage = np.sum(track_arrs[:, 2, :], axis=0)
        deletions = np.sum(track_arrs[:, 3, :], axis=0)
        
        n_positions = track_arrs.shape[2]
        match_prop = np.zeros(n_positions)
        mismatch_prop = np.zeros(n_positions)
        insertion_prop = np.zeros(n_positions)
        deletion_prop = np.zeros(n_positions)
        
        for i in range(n_positions):
            total = coverage[i] + insertions[i]
            if total > 0:
                match_prop[i] = matches[i] / total
                # Mismatches = coverage - matches - deletions
                mismatch_prop[i] = (coverage[i] - matches[i] - deletions[i]) / total
                deletion_prop[i] = deletions[i] / total
                insertion_prop[i] = insertions[i] / total
        
        return match_prop, mismatch_prop, insertion_prop, deletion_prop
    
    height_per = 5
    fig, axes = plt.subplots(len(present_trnas), 2, 
                            figsize=(16, len(present_trnas) * height_per))
    
    if len(present_trnas) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, trna in enumerate(present_trnas):
        # BWA
        if trna in bwa_data['by_trna']:
            track_arrs = bwa_data['by_trna'][trna]['track_arrs']
            n_reads = len(bwa_data['by_trna'][trna]['read_names'])
            
            match_prop, mismatch_prop, insertion_prop, deletion_prop = \
                compute_position_error_props(track_arrs)
            
            positions = np.arange(len(match_prop))
            
            axes[idx, 0].bar(positions, match_prop, label='Match', 
                           color='green', alpha=0.7, width=1.0)
            axes[idx, 0].bar(positions, mismatch_prop, bottom=match_prop, 
                           label='Mismatch', color='red', alpha=0.7, width=1.0)
            axes[idx, 0].bar(positions, deletion_prop, 
                           bottom=match_prop + mismatch_prop,
                           label='Deletion', color='purple', alpha=0.7, width=1.0)
            axes[idx, 0].bar(positions, insertion_prop, 
                           bottom=match_prop + mismatch_prop + deletion_prop,
                           label='Insertion', color='orange', alpha=0.7, width=1.0)
            
            axes[idx, 0].set_title(f'{trna} - BWA (N={n_reads:,})', fontsize=12)
            axes[idx, 0].set_ylabel('Proportion', fontsize=10)
            axes[idx, 0].set_ylim([0, 1.05])
            if idx == 0:
                axes[idx, 0].legend(loc='upper right', fontsize=8)
        
        # Zap
        if trna in zap_data['by_trna']:
            track_arrs = zap_data['by_trna'][trna]['track_arrs']
            n_reads = len(zap_data['by_trna'][trna]['read_names'])
            
            match_prop, mismatch_prop, insertion_prop, deletion_prop = \
                compute_position_error_props(track_arrs)
            
            positions = np.arange(len(match_prop))
            
            axes[idx, 1].bar(positions, match_prop, label='Match', 
                           color='green', alpha=0.7, width=1.0)
            axes[idx, 1].bar(positions, mismatch_prop, bottom=match_prop, 
                           label='Mismatch', color='red', alpha=0.7, width=1.0)
            axes[idx, 1].bar(positions, deletion_prop, 
                           bottom=match_prop + mismatch_prop,
                           label='Deletion', color='purple', alpha=0.7, width=1.0)
            axes[idx, 1].bar(positions, insertion_prop, 
                           bottom=match_prop + mismatch_prop + deletion_prop,
                           label='Insertion', color='orange', alpha=0.7, width=1.0)
            
            axes[idx, 1].set_title(f'{trna} - Zap (N={n_reads:,})', fontsize=12)
            axes[idx, 1].set_ylabel('Proportion', fontsize=10)
            axes[idx, 1].set_ylim([0, 1.05])
            if idx == 0:
                axes[idx, 1].legend(loc='upper right', fontsize=8)
    
    # Set x-labels only on bottom row
    axes[-1, 0].set_xlabel('Position', fontsize=10)
    axes[-1, 1].set_xlabel('Position', fontsize=10)
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}positional_error_barplots.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return present_trnas