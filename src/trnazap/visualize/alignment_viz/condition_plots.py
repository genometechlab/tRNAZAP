import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import json
from importlib.resources import files
plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def load_sort_order(model):
    """
    Load tRNA sort order from JSON file.
    
    Parameters
    ----------
    model : str
        'e_coli' or 'yeast'
    
    Returns
    -------
    list
        Ordered list of tRNA names
    """
    viz_path = files('trnazap').joinpath('visualize')
    
    if model == 'e_coli':
        json_path = viz_path / 'alignment_viz' / 'ecoli_label_sort_order.json'
    elif model == 'yeast':
        json_path = viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'
    else:
        # Default to yeast
        json_path = viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'
    
    with open(str(json_path), 'r') as infile:
        sort_order = json.load(infile)
    
    return sort_order


def apply_sort_order(df, sort_order, column='tRNA'):
    """
    Apply custom sort order to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
    sort_order : list
        Ordered list of values
    column : str
        Column name to sort by
    
    Returns
    -------
    pd.DataFrame
        Sorted DataFrame
    """
    # Only apply if the column exists
    if column not in df.columns:
        return df
    
    # Create a categorical type with the custom order
    df[column] = pd.Categorical(df[column], categories=sort_order, ordered=True)
    return df.sort_values(column)


# ============================================================================
# SINGLE CONDITION FIGURES
# ============================================================================

def plot_read_counts_per_trna(data_dict, condition_label, model, 
                              out_dir=None, out_prefix=''):
    """
    1. Bar plot of read counts for each tRNA class.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Extract counts
    trna_names = []
    counts = []
    for trna_name, data in data_dict.items():
        trna_names.append(trna_name)
        counts.append(len(data['track_arrs']))
    
    # Create DataFrame and apply custom sort
    df = pd.DataFrame({'tRNA': trna_names, 'Read Count': counts})
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width based on number of tRNAs
    n_trnas = len(df)
    fig_width = max(12, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Use bar plot with proper alignment
    x_positions = np.arange(len(df))
    ax.bar(x_positions, df['Read Count'], color='steelblue')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['tRNA'], rotation=90, ha='right')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Read Count', fontsize=12)
    ax.set_title(f'Read Counts per tRNA - {condition_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}read_counts_per_trna.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_identity_distribution(data_dict, condition_label, ident_threshold,
                               out_dir=None, out_prefix=''):
    """
    2. Histogram of identity scores across all reads.
    Binned at 2%, range from ident_threshold to 100%.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return np.array([])
    
    # Collect all identities and convert to percentage
    all_identities = []
    for data in data_dict.values():
        all_identities.extend(data['identities'])
    
    all_identities_pct = np.array(all_identities) * 100  # Convert to percentage
    
    # Plot with 2% bins
    fig, ax = plt.subplots(figsize=(8, 6))
    min_ident_pct = ident_threshold * 100
    bins = np.arange(min_ident_pct, 101, 2)  # ident_threshold to 100 in 2% increments
    sns.histplot(all_identities_pct, bins=bins, kde=False, ax=ax, color='coral')
    ax.set_xlabel('Identity (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Identity Distribution - {condition_label}', fontsize=14, fontweight='bold')
    ax.set_xlim(min_ident_pct, 100)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}identity_distribution.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.array(all_identities)


def plot_per_trna_identity_boxen(data_dict, condition_label, model, ident_threshold,
                                 out_dir=None, out_prefix=''):
    """
    3. Boxen plots of identity distributions for ALL tRNAs.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Prepare data for ALL tRNAs (sorted by sort_order)
    trna_names_sorted = [name for name in sort_order if name in data_dict]
    
    rows = []
    for trna_name in trna_names_sorted:
        for ident in data_dict[trna_name]['identities']:
            rows.append({'tRNA': trna_name, 'Identity': ident * 100})  # Convert to percentage
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(trna_names_sorted)
    fig_width = max(16, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.boxenplot(data=df, x='tRNA', y='Identity', ax=ax, color='lightblue')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Identity (%)', fontsize=12)
    ax.set_title(f'Identity Distribution per tRNA - {condition_label}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(ident_threshold * 100, 100)
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}per_trna_identity_boxen.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_coverage_heatmap(data_dict, condition_label, model,
                         out_dir=None, out_prefix=''):
    """
    4. Heatmap of coverage across all tRNA positions, normalized by max per row (0-100%), binned by 2%.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate mean coverage for each tRNA
    coverage_matrix = []
    trna_names = []
    
    for trna_name, data in data_dict.items():
        track_arrs = data['track_arrs']
        # Coverage is row 2 of track_arr
        mean_coverage = np.mean(track_arrs[:, 2, :], axis=0)
        coverage_matrix.append(mean_coverage)
        trna_names.append(trna_name)
    
    # Create DataFrame
    df = pd.DataFrame(coverage_matrix, index=trna_names)
    
    # Sort rows according to sort_order
    df = df.reindex([name for name in sort_order if name in df.index])
    
    # Normalize each row by its maximum value (scaled to 100%)
    df_normalized = df.div(df.max(axis=1), axis=0) * 100
    
    # Bin by 2%
    df_binned = (df_normalized // 2) * 2
    
    # Dynamic figure height
    n_trnas = len(df)
    fig_height = max(10, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, fig_height))
    sns.heatmap(df_binned, cmap='viridis', ax=ax, 
                vmin=0, vmax=100,
                cbar_kws={'label': 'Coverage (% of max)', 'ticks': np.arange(0, 101, 10)})
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('tRNA', fontsize=12)
    ax.set_title(f'Coverage Heatmap (Max-Normalized) - {condition_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}coverage_heatmap.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
        # Save TSV
        tsv_path = os.path.join(out_dir, f'{out_prefix}coverage_heatmap.tsv')
        df_normalized.to_csv(tsv_path, sep='\t')
    plt.close()
    
    return df_normalized


def plot_error_rate_heatmap_proportional(data_dict, condition_label, model,
                                        out_dir=None, out_prefix=''):
    """
    5. Proportional error rate heatmap (mismatches / coverage).
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate proportional error rates
    error_matrix = []
    trna_names = []
    
    for trna_name, data in data_dict.items():
        track_arrs = data['track_arrs']
        # Matches in row 0, coverage in row 2
        total_coverage = np.sum(track_arrs[:, 2, :], axis=0)
        total_matches = np.nansum(track_arrs[:, 0, :], axis=0)
        
        # Error rate = 1 - (matches / coverage)
        error_rate = np.zeros_like(total_coverage)
        mask = total_coverage > 0
        error_rate[mask] = 1 - (total_matches[mask] / total_coverage[mask])
        
        error_matrix.append(error_rate)
        trna_names.append(trna_name)
    
    # Create DataFrame
    df = pd.DataFrame(error_matrix, index=trna_names)
    
    # Sort rows according to sort_order
    df = df.reindex([name for name in sort_order if name in df.index])
    
    # Dynamic figure height
    n_trnas = len(df)
    fig_height = max(10, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, fig_height))
    sns.heatmap(df, cmap='Reds', vmin=0, vmax=0.2, ax=ax, 
                cbar_kws={'label': 'Error Rate'})
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('tRNA', fontsize=12)
    ax.set_title(f'Error Rate Heatmap (Proportional) - {condition_label}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}error_rate_heatmap_proportional.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
        # Save TSV
        tsv_path = os.path.join(out_dir, f'{out_prefix}error_rate_heatmap_proportional.tsv')
        df.to_csv(tsv_path, sep='\t')
    plt.close()
    
    return df


def plot_alignment_length_distribution(data_dict, condition_label, 
                                      out_dir=None, out_prefix=''):
    """
    6. Histogram of alignment lengths (discrete, no KDE).
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return np.array([])
    
    # Collect all alignment lengths
    all_lengths = []
    for data in data_dict.values():
        all_lengths.extend(data['alignment_lengths'])
    
    # Plot discrete histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(all_lengths, discrete=True, kde=False, ax=ax, color='seagreen')
    ax.set_xlabel('Alignment Length', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Alignment Length Distribution - {condition_label}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}alignment_length_distribution.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return np.array(all_lengths)


def plot_read_count_vs_identity(data_dict, condition_label, 
                                out_dir=None, out_prefix=''):
    """
    7. Scatter: read count (x, log scale) vs mean identity (y) per tRNA.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping plot.")
        return pd.DataFrame()
    
    # Prepare data
    trna_names = []
    read_counts = []
    mean_identities = []
    
    for trna_name, data in data_dict.items():
        trna_names.append(trna_name)
        read_counts.append(len(data['track_arrs']))
        mean_identities.append(np.mean(data['identities']) * 100)  # Convert to percentage
    
    df = pd.DataFrame({
        'tRNA': trna_names,
        'Read Count': read_counts,
        'Mean Identity': mean_identities
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Read Count', y='Mean Identity', ax=ax, 
                   s=80, alpha=0.6, color='purple')
    ax.set_xscale('log')
    ax.set_xlabel('Read Count (log scale)', fontsize=12)
    ax.set_ylabel('Mean Identity (%)', fontsize=12)
    ax.set_title(f'Read Count vs Identity - {condition_label}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}read_count_vs_identity.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def generate_trna_summary_table(data_dict, condition_label, model, out_dir=None, out_prefix=''):
    """
    Generate comprehensive summary table with one row per tRNA.
    """
    if not data_dict:
        print(f"No data available for {condition_label}, skipping summary table.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    rows = []
    for trna_name, data in data_dict.items():
        n_reads = len(data['track_arrs'])
        identities = data['identities']
        aln_lengths = data['alignment_lengths']
        
        row = {
            'tRNA': trna_name,
            'read_count': n_reads,
            'mean_identity': np.mean(identities),
            'std_identity': np.std(identities),
            'median_identity': np.median(identities),
            'min_identity': np.min(identities),
            'max_identity': np.max(identities),
            'mean_alignment_length': np.mean(aln_lengths),
            'std_alignment_length': np.std(aln_lengths),
            'median_alignment_length': np.median(aln_lengths)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    if out_dir:
        tsv_path = os.path.join(out_dir, f'{out_prefix}trna_summary.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Saved summary table: {tsv_path}")
    
    return df


# ============================================================================
# COMPARISON FIGURES
# ============================================================================

def plot_delta_read_percentage(dict_a, label_a, dict_b, label_b, model,
                               out_dir=None, out_prefix=''):
    """
    1. Delta % of total reads per tRNA class.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate total reads per condition
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNA names
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Calculate percentages and deltas
    rows = []
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        pct_a = (count_a / total_a) * 100
        pct_b = (count_b / total_b) * 100
        delta = pct_b - pct_a
        
        rows.append({
            'tRNA': trna,
            f'{label_a} %': pct_a,
            f'{label_b} %': pct_b,
            'Delta %': delta
        })
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(df)
    fig_width = max(12, n_trnas * 0.3)
    
    # Plot with aligned bars
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ['red' if x < 0 else 'blue' for x in df['Delta %']]
    x_positions = np.arange(len(df))
    ax.bar(x_positions, df['Delta %'], color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['tRNA'], rotation=90, ha='right')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel(f'Δ % of Reads ({label_b} - {label_a})', fontsize=12)
    ax.set_title(f'Delta Read Percentage: {label_b} vs {label_a}', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}delta_read_percentage.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_read_count_comparison_bars(dict_a, label_a, dict_b, label_b, model,
                                   out_dir=None, out_prefix=''):
    """
    2. Side-by-side bar plot for ALL tRNAs.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Get all tRNA names
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Prepare data
    rows = []
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        rows.append({'tRNA': trna, 'Condition': label_a, 'Count': count_a})
        rows.append({'tRNA': trna, 'Condition': label_b, 'Count': count_b})
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(all_trnas)
    fig_width = max(14, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.barplot(data=df, x='tRNA', y='Count', hue='Condition', ax=ax)
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Read Count', fontsize=12)
    ax.set_title(f'Read Count Comparison: {label_a} vs {label_b}', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Condition')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}read_count_comparison_bars.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_identity_comparison_boxen(dict_a, label_a, dict_b, label_b, model, ident_threshold,
                                   out_dir=None, out_prefix=''):
    """
    Boxen plots comparing identity distributions between two conditions.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Get all tRNA names
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    trna_names_sorted = [name for name in sort_order if name in all_trnas]
    
    # Prepare data
    rows = []
    for trna_name in trna_names_sorted:
        if trna_name in dict_a:
            for ident in dict_a[trna_name]['identities']:
                rows.append({'tRNA': trna_name, 'Condition': label_a, 'Identity': ident * 100})
        if trna_name in dict_b:
            for ident in dict_b[trna_name]['identities']:
                rows.append({'tRNA': trna_name, 'Condition': label_b, 'Identity': ident * 100})
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(trna_names_sorted)
    fig_width = max(16, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.boxenplot(data=df, x='tRNA', y='Identity', hue='Condition', ax=ax)
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel('Identity (%)', fontsize=12)
    ax.set_title(f'Identity Distribution Comparison: {label_a} vs {label_b}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(ident_threshold * 100, 100)
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Condition')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}identity_comparison_boxen.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_read_count_scatter_tpm(dict_a, label_a, dict_b, label_b, 
                                out_dir=None, out_prefix=''):
    """
    Scatter plot normalized for TPM (transcripts per million).
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Calculate TPM
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNAs
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Calculate TPMs
    trna_names = []
    tpm_a_list = []
    tpm_b_list = []
    
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        tpm_a = (count_a / total_a) * 1e6
        tpm_b = (count_b / total_b) * 1e6
        
        trna_names.append(trna)
        tpm_a_list.append(tpm_a)
        tpm_b_list.append(tpm_b)
    
    df = pd.DataFrame({
        'tRNA': trna_names,
        f'{label_a} TPM': tpm_a_list,
        f'{label_b} TPM': tpm_b_list
    })
    
    # Add pseudocount for log scale
    pseudocount = 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df, x=f'{label_a} TPM', y=f'{label_b} TPM', 
                   ax=ax, s=80, alpha=0.6)
    
    # Diagonal line
    max_val = max(df[f'{label_a} TPM'].max(), df[f'{label_b} TPM'].max())
    ax.plot([pseudocount, max_val], [pseudocount, max_val], 
            'k--', alpha=0.5, label='No change')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'{label_a} TPM (log scale)', fontsize=12)
    ax.set_ylabel(f'{label_b} TPM (log scale)', fontsize=12)
    ax.set_title(f'TPM Comparison: {label_a} vs {label_b}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}read_count_scatter_tpm.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_delta_tpm_absolute(dict_a, label_a, dict_b, label_b, model,
                            out_dir=None, out_prefix=''):
    """
    Histogram of absolute TPM differences (TPM_b - TPM_a).
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate TPM
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNAs
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Calculate deltas
    rows = []
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        tpm_a = (count_a / total_a) * 1e6
        tpm_b = (count_b / total_b) * 1e6
        delta_tpm = tpm_b - tpm_a
        
        rows.append({
            'tRNA': trna,
            f'{label_a} TPM': tpm_a,
            f'{label_b} TPM': tpm_b,
            'Delta TPM': delta_tpm
        })
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(df)
    fig_width = max(12, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ['red' if x < 0 else 'blue' for x in df['Delta TPM']]
    x_positions = np.arange(len(df))
    ax.bar(x_positions, df['Delta TPM'], color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['tRNA'], rotation=90, ha='right')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel(f'Δ TPM ({label_b} - {label_a})', fontsize=12)
    ax.set_title(f'Delta TPM (Absolute): {label_b} vs {label_a}', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}delta_tpm_absolute.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_delta_tpm_log2fc(dict_a, label_a, dict_b, label_b, model,
                          out_dir=None, out_prefix=''):
    """
    Histogram of log2 fold change of TPMs.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate TPM
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNAs
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Calculate log2FC
    rows = []
    pseudocount = 1
    
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        tpm_a = (count_a / total_a) * 1e6
        tpm_b = (count_b / total_b) * 1e6
        
        log2fc = np.log2((tpm_b + pseudocount) / (tpm_a + pseudocount))
        
        rows.append({
            'tRNA': trna,
            f'{label_a} TPM': tpm_a,
            f'{label_b} TPM': tpm_b,
            'log2FC': log2fc
        })
    
    df = pd.DataFrame(rows)
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    # Dynamic figure width
    n_trnas = len(df)
    fig_width = max(12, n_trnas * 0.3)
    
    # Plot
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    colors = ['red' if x < 0 else 'blue' for x in df['log2FC']]
    x_positions = np.arange(len(df))
    ax.bar(x_positions, df['log2FC'], color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df['tRNA'], rotation=90, ha='right')
    ax.set_xlabel('tRNA', fontsize=12)
    ax.set_ylabel(f'log₂(FC) [{label_b}/{label_a}]', fontsize=12)
    ax.set_title(f'log₂ Fold Change (TPM): {label_b} vs {label_a}', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}delta_tpm_log2fc.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_volcano(dict_a, label_a, dict_b, label_b, 
                 fc_threshold=1.5, pval_threshold=0.05, 
                 out_dir=None, out_prefix=''):
    """
    Volcano plot: log2FC calculated on TPMs, p-values from chi-square on counts.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping plot.")
        return pd.DataFrame()
    
    # Calculate totals for chi-square
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNAs
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    # Statistical tests
    trna_names = []
    log2_fcs = []
    pvals = []
    tpm_a_list = []
    tpm_b_list = []
    
    pseudocount = 1
    
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        # Calculate TPMs for log2FC
        tpm_a = (count_a / total_a) * 1e6
        tpm_b = (count_b / total_b) * 1e6
        log2_fc = np.log2((tpm_b + pseudocount) / (tpm_a + pseudocount))
        
        # Chi-square test on counts
        observed = np.array([count_a, count_b])
        expected = np.array([total_a, total_b]) * (count_a + count_b) / (total_a + total_b)
        
        # Avoid zero expected values
        if np.any(expected == 0):
            pval = 1.0
        else:
            chi2, pval = stats.chisquare(observed, expected)
        
        trna_names.append(trna)
        log2_fcs.append(log2_fc)
        pvals.append(pval)
        tpm_a_list.append(tpm_a)
        tpm_b_list.append(tpm_b)
    
    # FDR correction
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    
    df = pd.DataFrame({
        'tRNA': trna_names,
        f'{label_a} TPM': tpm_a_list,
        f'{label_b} TPM': tpm_b_list,
        'log2FC': log2_fcs,
        'pval': pvals,
        'qval': qvals,
        '-log10(pval)': -np.log10(np.array(pvals) + 1e-300)  # Avoid log(0)
    })
    
    # Classify significance
    df['Significant'] = ((np.abs(df['log2FC']) > np.log2(fc_threshold)) & 
                         (df['pval'] < pval_threshold))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Non-significant
    nonsig = df[~df['Significant']]
    sig = df[df['Significant']]
    
    ax.scatter(nonsig['log2FC'], nonsig['-log10(pval)'], 
              c='gray', alpha=0.5, s=40, label='Not significant')
    ax.scatter(sig['log2FC'], sig['-log10(pval)'], 
              c='red', alpha=0.7, s=60, label='Significant')
    
    # Threshold lines
    ax.axhline(-np.log10(pval_threshold), color='blue', linestyle='--', 
              alpha=0.5, label=f'p = {pval_threshold}')
    ax.axvline(np.log2(fc_threshold), color='green', linestyle='--', 
              alpha=0.5, label=f'FC = {fc_threshold}')
    ax.axvline(-np.log2(fc_threshold), color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel(f'log₂(Fold Change) [{label_b}/{label_a}] (TPM)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title(f'Volcano Plot: {label_b} vs {label_a}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}volcano_plot.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def plot_per_position_error_deltas(dict_a, label_a, dict_b, label_b, 
                                   trna_name, out_dir=None, out_prefix=''):
    """
    Per-position delta plot with separate bars for mismatches, deletions, insertions.
    """
    if trna_name not in dict_a or trna_name not in dict_b:
        print(f"tRNA {trna_name} not found in both conditions")
        return None
    
    # Get data
    tracks_a = dict_a[trna_name]['track_arrs']
    tracks_b = dict_b[trna_name]['track_arrs']
    
    n_positions = tracks_a.shape[2]
    
    # Calculate rates for condition A
    # track_arr structure: [0]=matches, [1]=insertions, [2]=coverage, [3]=deletions
    coverage_a = np.sum(tracks_a[:, 2, :], axis=0)
    matches_a = np.nansum(tracks_a[:, 0, :], axis=0)
    insertions_a = np.sum(tracks_a[:, 1, :], axis=0)
    deletions_a = np.sum(tracks_a[:, 3, :], axis=0)
    
    mismatch_rate_a = np.zeros(n_positions)
    insertion_rate_a = np.zeros(n_positions)
    deletion_rate_a = np.zeros(n_positions)
    
    mask_a = coverage_a > 0
    mismatch_rate_a[mask_a] = (coverage_a[mask_a] - matches_a[mask_a]) / coverage_a[mask_a]
    insertion_rate_a[mask_a] = insertions_a[mask_a] / coverage_a[mask_a]
    deletion_rate_a[mask_a] = deletions_a[mask_a] / coverage_a[mask_a]
    
    # Calculate rates for condition B
    coverage_b = np.sum(tracks_b[:, 2, :], axis=0)
    matches_b = np.nansum(tracks_b[:, 0, :], axis=0)
    insertions_b = np.sum(tracks_b[:, 1, :], axis=0)
    deletions_b = np.sum(tracks_b[:, 3, :], axis=0)
    
    mismatch_rate_b = np.zeros(n_positions)
    insertion_rate_b = np.zeros(n_positions)
    deletion_rate_b = np.zeros(n_positions)
    
    mask_b = coverage_b > 0
    mismatch_rate_b[mask_b] = (coverage_b[mask_b] - matches_b[mask_b]) / coverage_b[mask_b]
    insertion_rate_b[mask_b] = insertions_b[mask_b] / coverage_b[mask_b]
    deletion_rate_b[mask_b] = deletions_b[mask_b] / coverage_b[mask_b]
    
    # Calculate deltas
    delta_mismatch = mismatch_rate_b - mismatch_rate_a
    delta_insertion = insertion_rate_b - insertion_rate_a
    delta_deletion = deletion_rate_b - deletion_rate_a
    
    # Prepare DataFrame
    positions = np.arange(n_positions)
    rows = []
    for i in positions:
        rows.append({'Position': i, 'Error Type': 'Mismatch', 'Delta': delta_mismatch[i]})
        rows.append({'Position': i, 'Error Type': 'Insertion', 'Delta': delta_insertion[i]})
        rows.append({'Position': i, 'Error Type': 'Deletion', 'Delta': delta_deletion[i]})
    
    df = pd.DataFrame(rows)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=df, x='Position', y='Delta', hue='Error Type', ax=ax,
               palette={'Mismatch': 'orange', 'Insertion': 'green', 'Deletion': 'purple'})
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel(f'Δ Error Rate ({label_b} - {label_a})', fontsize=12)
    ax.set_title(f'Per-Position Error Rate Deltas: {trna_name}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Error Type')
    plt.tight_layout()
    
    if out_dir:
        save_path = os.path.join(out_dir, f'{out_prefix}per_position_error_deltas_{trna_name}.pdf')
        fig.get_figure().savefig(save_path, dpi=300, bbox_inches='tight')
        # Save TSV
        tsv_path = os.path.join(out_dir, f'{out_prefix}per_position_error_deltas_{trna_name}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
    plt.close()
    
    return df


def generate_comparison_summary_table(dict_a, label_a, dict_b, label_b, model, 
                                      fc_threshold=1.5, pval_threshold=0.05,
                                      out_dir=None, out_prefix=''):
    """
    Generate comprehensive comparison summary table with one row per tRNA.
    """
    if not dict_a or not dict_b:
        print(f"Insufficient data for comparison, skipping summary table.")
        return pd.DataFrame()
    
    # Load sort order
    sort_order = load_sort_order(model)
    
    # Calculate totals
    total_a = sum(len(data['track_arrs']) for data in dict_a.values())
    total_b = sum(len(data['track_arrs']) for data in dict_b.values())
    
    # Get all tRNAs
    all_trnas = set(dict_a.keys()) | set(dict_b.keys())
    
    rows = []
    pseudocount = 1
    
    for trna in all_trnas:
        count_a = len(dict_a[trna]['track_arrs']) if trna in dict_a else 0
        count_b = len(dict_b[trna]['track_arrs']) if trna in dict_b else 0
        
        # Percentages
        pct_a = (count_a / total_a) * 100
        pct_b = (count_b / total_b) * 100
        delta_pct = pct_b - pct_a
        
        # TPMs
        tpm_a = (count_a / total_a) * 1e6
        tpm_b = (count_b / total_b) * 1e6
        delta_tpm = tpm_b - tpm_a
        
        # Log2FC
        log2fc = np.log2((tpm_b + pseudocount) / (tpm_a + pseudocount))
        
        # Chi-square test
        observed = np.array([count_a, count_b])
        expected = np.array([total_a, total_b]) * (count_a + count_b) / (total_a + total_b)
        
        if np.any(expected == 0):
            pval = 1.0
        else:
            chi2, pval = stats.chisquare(observed, expected)
        
        # Identity stats if available
        mean_ident_a = np.mean(dict_a[trna]['identities']) if trna in dict_a else np.nan
        mean_ident_b = np.mean(dict_b[trna]['identities']) if trna in dict_b else np.nan
        
        row = {
            'tRNA': trna,
            f'{label_a}_count': count_a,
            f'{label_b}_count': count_b,
            f'{label_a}_pct': pct_a,
            f'{label_b}_pct': pct_b,
            'delta_pct': delta_pct,
            f'{label_a}_TPM': tpm_a,
            f'{label_b}_TPM': tpm_b,
            'delta_TPM': delta_tpm,
            'log2FC': log2fc,
            'fold_change': 2 ** log2fc,
            'pval': pval,
            f'{label_a}_mean_identity': mean_ident_a,
            f'{label_b}_mean_identity': mean_ident_b
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # FDR correction
    _, qvals, _, _ = multipletests(df['pval'].values, method='fdr_bh')
    df['qval'] = qvals
    
    # Significance flag
    df['significant'] = ((np.abs(df['log2FC']) > np.log2(fc_threshold)) & 
                         (df['pval'] < pval_threshold))
    
    df = apply_sort_order(df, sort_order, 'tRNA')
    
    if out_dir:
        tsv_path = os.path.join(out_dir, f'{out_prefix}comparison_summary.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Saved comparison summary table: {tsv_path}")
    
    return df