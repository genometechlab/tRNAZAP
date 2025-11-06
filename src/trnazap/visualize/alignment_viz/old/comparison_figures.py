from .read_pass_for_classification import ref_conversion, multiprocess_trna_data, merge_multiprocess
import pysam
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import multiprocessing
from matplotlib.colors import LogNorm
from importlib.resources import files
import pickle
import json
import numpy as np
from scipy import stats
from time import time

plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')
sns.set(rc={'figure.figsize': (11.7, 8.27)})

def suppress_plotting_warnings():
    """
    Suppress common matplotlib and numpy warnings during plotting.
    Call this at the beginning of create_figures().
    """
    # Suppress matplotlib UserWarnings about tick labels
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Suppress numpy RuntimeWarnings about division
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

def paired_identity_test(bwa_idents, zap_idents, margin=0.00, test_name=""):
    """
    Core statistical test comparing paired identity scores.
    One-sided test: Is Zap superior to BWA by at least margin?
    
    Parameters:
    -----------
    bwa_idents : np.array
        BWA identity scores
    zap_idents : np.array  
        Zap identity scores (paired with bwa_idents)
    margin : float
        Margin for superiority test (default 0.04 = 4%)
    test_name : str
        Descriptive name for the comparison
        
    Returns:
    --------
    dict with keys: test_name, n, mean_diff, median_diff, p_value, significant
    """
    differences = zap_idents - bwa_idents
    
    # One-sided superiority test: Is Zap better by at least margin?
    w_stat, p_value = stats.wilcoxon(differences - margin, alternative='greater')
    
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    significant = p_value < 0.05
    
    # Print summary
    if test_name:
        print(f"\n{test_name}")
    print(f"  N={len(differences):,} | Mean Δ={mean_diff:+.4f} | Median Δ={median_diff:+.4f} | p={p_value:.4e} | {'✓ Zap superior' if significant else '✗ Not superior'}")
    
    return {
        'test_name': test_name,
        'n': len(differences),
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'p_value': p_value,
        'significant': significant,
        'w_statistic': w_stat,
        'margin': margin
    }


def test_matched_reads_superiority(read_ident_dict, id_set):
    """
    Test if Zap is superior to BWA by ≥4% for reads aligned by BOTH aligners.
    
    Parameters:
    -----------
    read_ident_dict : dict
        Dictionary with read identities
    id_set : set
        Set of read IDs successfully aligned by both aligners
        
    Returns:
    --------
    dict with test results
    """
    zap_idents = []
    bwa_idents = []
    
    for read_id in id_set:
        if (read_ident_dict[read_id]['zap'] is not None and 
            read_ident_dict[read_id]['bwa'] is not None):
            zap_idents.append(read_ident_dict[read_id]['zap'])
            bwa_idents.append(read_ident_dict[read_id]['bwa'])
    
    zap_idents = np.array(zap_idents)
    bwa_idents = np.array(bwa_idents)
    
    return paired_identity_test(
        bwa_idents, 
        zap_idents, 
        margin=0.00,
        test_name="Reads aligned by BOTH aligners"
    )


def test_all_reads_superiority(read_ident_dict):
    """
    Test if Zap is superior to BWA for ALL reads with valid data.
    
    Parameters:
    -----------
    read_ident_dict : dict
        Dictionary with read identities
        
    Returns:
    --------
    dict with test results
    """
    zap_idents = []
    bwa_idents = []
    
    for read_id, read_data in read_ident_dict.items():
        if (read_data['zap'] is not None and 
            read_data['bwa'] is not None):
            zap_idents.append(read_data['zap'])
            bwa_idents.append(read_data['bwa'])
    
    zap_idents = np.array(zap_idents)
    bwa_idents = np.array(bwa_idents)
    
    return paired_identity_test(
        bwa_idents, 
        zap_idents, 
        margin=0.00,
        test_name="ALL reads with valid identities"
    )


def test_misclassified_reads(read_ident_dict, mismatch_id_set):
    """
    Test if Zap has higher identity for reads where aligners DISAGREE on tRNA classification.
    
    Parameters:
    -----------
    read_ident_dict : dict
        Dictionary with read identities
    mismatch_id_set : set
        Set of read IDs where BWA and Zap disagree on classification
        
    Returns:
    --------
    dict with test results
    """
    zap_idents = []
    bwa_idents = []
    
    for read_id in mismatch_id_set:
        if (read_ident_dict[read_id]['zap'] is not None and 
            read_ident_dict[read_id]['bwa'] is not None):
            zap_idents.append(read_ident_dict[read_id]['zap'])
            bwa_idents.append(read_ident_dict[read_id]['bwa'])
    
    zap_idents = np.array(zap_idents)
    bwa_idents = np.array(bwa_idents)
    
    return paired_identity_test(
        bwa_idents, 
        zap_idents, 
        margin=0.00,
        test_name="Misclassified reads (aligners disagree on tRNA class)"
    )


def save_statistical_results(results_dict, out_pre, out_dir):
    """
    Save statistical test results to CSV file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary of test results
    out_pre : str
        Output file prefix
    out_dir : str
        Output directory
    """
    # Convert results to DataFrame
    rows = []
    for key, result in results_dict.items():
        rows.append({
            'Test': result['test_name'],
            'N_reads': result['n'],
            'Mean_diff': result['mean_diff'],
            'Median_diff': result['median_diff'],
            'P_value': result['p_value'],
            'Significant': result['significant'],
            'W_statistic': result['w_statistic'],
            'Margin': result['margin']
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_path = os.path.join(out_dir, f"{out_pre}_statistical_tests.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')


def run_statistical_comparisons(read_ident_dict, id_set, mismatch_id_set, out_pre, out_dir):
    """
    Run complete statistical comparison suite and save results.
    
    Parameters:
    -----------
    read_ident_dict : dict
        Dictionary with read identities
    id_set : set
        Set of read IDs successfully aligned by both aligners
    mismatch_id_set : set
        Set of read IDs where aligners disagree on classification
    out_pre : str
        Output file prefix
    out_dir : str
        Output directory
    """
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: Zap vs BWA Identity (One-sided)")
    print("="*70)
    
    # Test 1: Matched reads
    result_matched = test_matched_reads_superiority(read_ident_dict, id_set)
    
    # Test 2: All reads
    result_all = test_all_reads_superiority(read_ident_dict)
    
    # Test 3: Misclassified reads
    result_mismatch = test_misclassified_reads(read_ident_dict, mismatch_id_set)
    
    print("="*70 + "\n")
    
    # Compile results
    results = {
        'matched_reads': result_matched,
        'all_reads': result_all,
        'misclassified_reads': result_mismatch
    }
    
    # Save to file
    save_statistical_results(results, out_pre, out_dir)
    
    return results

def per_class_identity_plot(in_df, out_pre, out_dir, sort_order):
    color_dict = {
        'bwa': 'b',
        'zap': 'orange'
    }
    
    # Filter sort_order to only include categories present in the data
    present_categories = [cat for cat in sort_order if cat in in_df['trna'].values]
    
    fig, ax = plt.subplots(1)
    fig.set_figheight(6)
    fig.set_figwidth(8 * len(present_categories) * 0.3)
    sns.boxenplot(data=in_df, x='trna', y='ident', hue='aligner', palette=color_dict, order=present_categories)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_per_class_identity.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def class_count_plots(count_df, out_pre, out_dir, sort_order):
    # Filter sort_order to only include categories present in the data
    present_categories = [cat for cat in sort_order if cat in count_df['trna'].values]
    
    fig, ax = plt.subplots(1)
    fig.set_figheight(10)
    fig.set_figwidth(32)
    sns.barplot(data=count_df, x='trna', y='count', hue='aligner', order=present_categories)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_class_counts.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def class_delta_plots(deltas, delta_label, out_pre, out_dir, sort_order):
    # Create mapping for custom sort order
    order_dict = {label: i for i, label in enumerate(sort_order)}
    
    # Sort by position in sort_order (items not in sort_order go to end)
    sorted_pairs = sorted(zip(delta_label, deltas), 
                         key=lambda x: order_dict.get(x[0], float('inf')))
    
    # Unzip back into separate lists
    sorted_labels, sorted_deltas = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    fig, ax = plt.subplots(1)
    fig.set_figwidth(14)
    sns.barplot(y=list(sorted_deltas), x=list(range(len(sorted_deltas))))
    plt.xticks(range(len(sorted_deltas)), sorted_labels, rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_deltas_class_counts.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def position_ident_plot(zap_position_ident_dict,
                        bwa_position_ident_dict,
                        out_pre,
                        out_dir,
                        sort_order):
    
    # Filter sort_order to only include tRNAs present in the data
    present_trnas = [trna for trna in sort_order if trna in zap_position_ident_dict]
    
    height_per = 5
    fig, ax = plt.subplots(len(present_trnas), 1)
    fig.set_figheight(len(present_trnas) * height_per)
    
    # Create index mapping based on sort_order
    t_to_idx = {t: i for i, t in enumerate(present_trnas)}
    
    for trna in present_trnas:
        if trna in zap_position_ident_dict:
            i = t_to_idx[trna]
            ident, insertions, coverage, align_len = zap_position_ident_dict[trna]
            ax[i].set_title(trna)
            ident = safe_divide_for_plot(ident, coverage, insertions)
            sns.lineplot(ident, ax=ax[i], color='orange', drawstyle='steps-mid', label="zap", legend=False)
    
    for trna in present_trnas:
        if trna in bwa_position_ident_dict:
            i = t_to_idx[trna]
            ident, insertions, coverage, align_len = bwa_position_ident_dict[trna]
            ident = safe_divide_for_plot(ident, coverage, insertions)
            sns.lineplot(ident, ax=ax[i], color='blue', drawstyle='steps-mid', linestyle="--", label="bwa", legend=False)
    
    ax[0].legend(loc="best")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_positional_identity.pdf")
    fig.savefig(out_path)
    plt.close(fig)


def plot_alignment_heatmap(count_matrix,
                           out_pre,
                           out_dir,
                           sort_order):
    
    # Filter sort_order for rows/columns present in the matrix
    present_rows = [cat for cat in sort_order if cat in count_matrix.index]
    present_cols = [cat for cat in sort_order if cat in count_matrix.columns]
    
    # Reindex the matrix to match sort_order
    count_matrix = count_matrix.reindex(index=present_rows, columns=present_cols, fill_value=0)
    
    fig, ax = plt.subplots(1)
    fig.set_figheight(max(12, len(present_rows) * 0.3))  # Scale height with number of rows
    fig.set_figwidth(max(14, len(present_cols) * 0.3))   # Scale width with number of cols
    sns.heatmap(count_matrix, 
                norm=LogNorm(), 
                cmap='viridis', 
                cbar=True, 
                xticklabels=present_cols, 
                yticklabels=present_rows,
                linewidths=0, 
                linecolor='none',
                square=True
               )
    
    ax.set_facecolor('white')
    ax.set_xticks(np.arange(len(present_cols))+0.5)
    ax.set_xticklabels(present_cols, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(present_rows))+0.5)
    ax.set_yticklabels(present_rows, rotation=0)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_alignment_classification_heatmap.pdf")
    fig.savefig(out_path)
    plt.close(fig)

def plot_one_aligner_hist_plots(bwa_no_zap_ident, zap_no_bwa_ident, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    if len(bwa_no_zap_ident) > 1:
        sns.histplot(bwa_no_zap_ident, binwidth=0.025, label="bwa", color='b', alpha=0.75)
    if len(zap_no_bwa_ident) > 1:
        sns.histplot(zap_no_bwa_ident, binwidth=0.025, label="zap", color='orange', alpha=0.75)
    plt.legend(loc='best')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_one_aligner_only_ident_hist_plot.pdf")
    fig.savefig(out_path)


def plot_per_read_miss_classified_ident(misclassified_ident_df, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    sns.histplot(data=misclassified_ident_df, x='zap_ident', y='bwa_ident', binwidth=0.025)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_misclassified_read_ident.pdf")
    fig.savefig(out_path)


def plot_per_read_ident(read_ident_df, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    sns.histplot(data=read_ident_df, x='zap_ident', y='bwa_ident', binwidth=0.0125)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_pre_read_ident.pdf")
    fig.savefig(out_path)

def plot_hexbin_delta(read_ident_dict, id_set, out_pre, out_dir, gridsize=30, save_data=False):
    """
    Create hexbin plots showing the delta between bwa and zap alignments.
    Uses the read_ident_dict that now contains alignment lengths.
    
    Parameters:
    -----------
    read_ident_dict : dict
        Dictionary with read IDs as keys, containing:
        {'bwa': identity, 'zap': identity, 'bwa_len': length, 'zap_len': length}
    id_set : set
        Set of read IDs that were successfully aligned by both aligners (not used in filtering)
    out_pre : str
        Output file prefix
    out_dir : str
        Output directory
    gridsize : int
        Number of hexagons in x-direction (default: 30)
    save_data : bool
        Whether to save hexbin data to CSV
    """
    # Extract alignment data from read_ident_dict
    # Include ALL reads that each aligner successfully aligned, not just shared ones
    bwa_lengths = []
    bwa_idents = []
    zap_lengths = []
    zap_idents = []
    
    for read_id in read_ident_dict:
        # Include reads where BWA has valid data
        if (read_ident_dict[read_id]['bwa'] is not None and 
            read_ident_dict[read_id]['bwa_len'] is not None):
            bwa_lengths.append(read_ident_dict[read_id]['bwa_len'])
            bwa_idents.append(read_ident_dict[read_id]['bwa'])
        
        # Include reads where Zap has valid data
        if (read_ident_dict[read_id]['zap'] is not None and 
            read_ident_dict[read_id]['zap_len'] is not None):
            zap_lengths.append(read_ident_dict[read_id]['zap_len'])
            zap_idents.append(read_ident_dict[read_id]['zap'])
    
    # Convert to numpy arrays
    bwa_lengths = np.array(bwa_lengths)
    bwa_idents = np.array(bwa_idents)
    zap_lengths = np.array(zap_lengths)
    zap_idents = np.array(zap_idents)
        
    if len(bwa_lengths) == 0:
        print("No valid reads found for hexbin plot")
        return None, None
    
    # Combine all data to get consistent extent
    all_lengths = np.concatenate([bwa_lengths, zap_lengths])
    all_idents = np.concatenate([bwa_idents, zap_idents])
    
    # Define consistent extent for both plots
    length_min, length_max = all_lengths.min(), all_lengths.max()
    ident_min, ident_max = all_idents.min(), all_idents.max()
    extent = [length_min, length_max, ident_min, ident_max]
    
    # Create indicator arrays: 1 for BWA, 0 for Zap (for counting)
    # We'll create hexbins using all combined data, then use C parameter to count separately
    bwa_indicator = np.ones(len(bwa_lengths))
    zap_indicator = np.ones(len(zap_lengths))
    
    # Combine all data with indicators
    combined_lengths = np.concatenate([bwa_lengths, zap_lengths])
    combined_idents = np.concatenate([bwa_idents, zap_idents])
    aligner_indicator = np.concatenate([bwa_indicator, np.zeros(len(zap_lengths))])
    
    # Create base hexbin structure using ALL data to ensure consistent grid
    fig_base, ax_base = plt.subplots(1, figsize=(8, 6))
    hb_base = ax_base.hexbin(combined_lengths, combined_idents,
                             gridsize=gridsize, extent=extent,
                             mincnt=1)
    base_offsets = hb_base.get_offsets()
    plt.close(fig_base)
    
    # Now create BWA hexbin using the combined data structure but counting only BWA points
    fig_bwa, ax_bwa = plt.subplots(1, figsize=(8, 6))
    hb_bwa = ax_bwa.hexbin(combined_lengths, combined_idents,
                           C=aligner_indicator,
                           reduce_C_function=np.sum,
                           gridsize=gridsize, extent=extent, 
                           cmap='Blues', mincnt=0)
    ax_bwa.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax_bwa.set_ylabel('Identity', fontsize=12)
    ax_bwa.set_title('BWA Alignments', fontsize=14)
    plt.colorbar(hb_bwa, ax=ax_bwa, label='Count')
    plt.tight_layout()
    out_path_bwa = os.path.join(out_dir, f"{out_pre}_hexbin_bwa_length_identity.pdf")
    fig_bwa.savefig(out_path_bwa, dpi=300, bbox_inches='tight')
    plt.close(fig_bwa)
    
    # Create Zap hexbin using the combined data structure but counting only Zap points
    fig_zap, ax_zap = plt.subplots(1, figsize=(8, 6))
    hb_zap = ax_zap.hexbin(combined_lengths, combined_idents,
                           C=1 - aligner_indicator,  # 1 for Zap, 0 for BWA
                           reduce_C_function=np.sum,
                           gridsize=gridsize, extent=extent, 
                           cmap='Oranges', mincnt=0)
    ax_zap.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax_zap.set_ylabel('Identity', fontsize=12)
    ax_zap.set_title('tRNA-zap Alignments', fontsize=14)
    plt.colorbar(hb_zap, ax=ax_zap, label='Count')
    plt.tight_layout()
    out_path_zap = os.path.join(out_dir, f"{out_pre}_hexbin_zap_length_identity.pdf")
    fig_zap.savefig(out_path_zap, dpi=300, bbox_inches='tight')
    plt.close(fig_zap)
    
    # Extract counts and calculate delta
    counts_bwa = hb_bwa.get_array()
    counts_zap = hb_zap.get_array()
    
    
    delta = counts_zap - counts_bwa
    
    # Plot 3: Delta hexbin
    fig_delta, ax_delta = plt.subplots(1, figsize=(8, 6))
    # Use the same combined data structure for the delta plot
    hb_delta = ax_delta.hexbin(combined_lengths, combined_idents,
                               C=np.zeros_like(combined_lengths),
                               reduce_C_function=np.sum,
                               gridsize=gridsize, extent=extent,
                               mincnt=0)
    hb_delta.set_array(delta)
    hb_delta.set_cmap('RdBu_r')
    
    # Set symmetric color limits
    vmax = np.abs(delta).max()
    hb_delta.set_clim(-vmax, vmax)
    
    ax_delta.set_xlabel('Alignment Length (bp)', fontsize=12)
    ax_delta.set_ylabel('Identity', fontsize=12)
    ax_delta.set_title('Delta (Zap - BWA)', fontsize=14)
    plt.colorbar(hb_delta, ax=ax_delta, label='Count Difference')
    plt.tight_layout()
    out_path_delta = os.path.join(out_dir, f"{out_pre}_hexbin_delta_length_identity.pdf")
    fig_delta.savefig(out_path_delta, dpi=300, bbox_inches='tight')
    plt.close(fig_delta)
    
    return delta, hb_delta.get_offsets()


def per_read_output(bwamem,
                    trnazap,
                    bwa_mem_ref_dict,
                    bwa_ref_lens,
                    zap_ref_dict,
                    zap_ref_lens,
                    threads):

    args_list = [[bwamem, trnazap, bwa_mem_ref_dict, bwa_ref_lens, zap_ref_dict, zap_ref_lens, int(threads), i] for i in
                 range(int(threads))]

    with multiprocessing.Pool(processes=int(threads)) as p:
        outputs = p.map(multiprocess_trna_data, args_list)

    return merge_multiprocess(outputs)


def conversion_dicts(bwa_ref, zap_ref):

    viz_path = files('trnazap').joinpath('visualize')
    with open(str(viz_path / 'alignment_viz' / 'align_to_viz_labels.json'), 'r') as infile:
        ref_label_dict = json.load(infile)
    
    bwa_mem_ref_dict = {}
    zap_ref_dict = {}

    for seq in pysam.FastxFile(bwa_ref):
        bwa_mem_ref_dict[ref_label_dict[seq.name]] = seq.sequence
    for seq in pysam.FastxFile(zap_ref):
        zap_ref_dict[ref_label_dict[seq.name]] = seq.sequence

    ref_convert = ref_conversion(bwa_ref,
                                 zap_ref)

    bwa_ref_lens = {}
    for key in bwa_mem_ref_dict:
        bwa_ref_lens[key] = len(bwa_mem_ref_dict[key])
    zap_ref_lens = {}
    for key in zap_ref_dict:
        zap_ref_lens[key] = len(zap_ref_dict[key])

    return (ref_convert,
            bwa_mem_ref_dict,
            zap_ref_dict,
            bwa_ref_lens,
            zap_ref_lens)


def alignment_counts(total_read_dict):
    bwa_alignments = {}
    zap_alignments = {}

    for key in total_read_dict:
        assert 'zap_to_bwa' in total_read_dict[key], f"{total_read_dict[key]}" 
        zap = total_read_dict[key]['zap_to_bwa']
        if zap not in zap_alignments:
            zap_alignments[zap] = 0
        zap_alignments[zap] += 1

        bwa = total_read_dict[key]['bwa']
        if bwa not in bwa_alignments:
            bwa_alignments[bwa] = 0
        bwa_alignments[bwa] += 1

    both = 0
    zap_fail = 0
    bwa_fail = 0

    for read in total_read_dict:
        b = total_read_dict[read]['bwa'] == "Failed"
        z = total_read_dict[read]['zap'] == "Failed"
        if b and z:
            both += 1
        elif b:
            bwa_fail += 1
        elif z:
            zap_fail += 1

    count_df = pd.DataFrame([
                                {'trna': key, 'count': value, 'aligner': 'bwa'}
                                for key, value in bwa_alignments.items()]
                            + [
                                {'trna': key, 'count': value, 'aligner': 'zap'}
                                for key, value in zap_alignments.items()])

    count_df.loc[count_df["trna"] == "ped", "trna"] = "Unmapped"
    count_df.loc[count_df["trna"] == "d", "trna"] = "Failed"
    count_df.loc[count_df["trna"] == 'NA', "trna"] = "No trna"

    return count_df


def make_classification_df():
    classification_df = pd.DataFrame()
    return classification_df

def count_deltas(count_df):
    deltas = []
    delta_label = []
    for t in set(count_df['trna'].to_list()):
        if t in ['Unmapped', 'Failed', 'No trna']:
            continue
        sub_df = count_df[count_df['trna'] == t]
        if len(sub_df[sub_df['aligner'] == 'zap']['count'].to_list()) == 0:
            zap = 0
        else:
            if len(sub_df[sub_df['aligner'] == 'zap']['count'].to_list()) > 0:
                zap = sub_df[sub_df['aligner'] == 'zap']['count'].to_list()[0]
            else:
                zap = 0
        if len(sub_df[sub_df['aligner'] == 'bwa']['count'].to_list()) > 0:
            bwa = sub_df[sub_df['aligner'] == 'bwa']['count'].to_list()[0]
        else:
            bwa = 0
        deltas.append(zap - bwa)
        delta_label.append(t)
    return deltas, delta_label

def calculate_error_proportions(track_arr):
    """
    Calculate total error counts and proportions across all positions.
    Used for summary statistics table.
    
    Parameters:
    -----------
    track_arr : np.array
        Shape (4, length) where:
        [0] = matches at each position
        [1] = insertions at each position  
        [2] = coverage at each position (1 if covered, 0 if not)
        [3] = deletions at each position
    
    Returns:
    --------
    dict with keys: matches, mismatches, insertions, deletions (all as counts)
    """
    matches = np.nansum(track_arr[0])
    insertions = np.nansum(track_arr[1])
    coverage = np.nansum(track_arr[2])
    deletions = np.nansum(track_arr[3])
    
    # Mismatches = positions that were covered but didn't match and weren't deletions
    # At each position: if coverage > 0 and not NaN, then mismatches = coverage - matches - deletions
    mismatches = 0
    for i in range(track_arr.shape[1]):
        if track_arr[2][i] > 0 and not np.isnan(track_arr[0][i]):
            # This position was covered
            mismatches += (track_arr[2][i] - track_arr[0][i] - track_arr[3][i])
    
    return {
        'matches': matches,
        'mismatches': mismatches,
        'insertions': insertions,
        'deletions': deletions
    }


def compute_position_error_props(track_arr):
    """
    Compute error proportions at each position.
    Used for positional error bar plots.
    
    Parameters:
    -----------
    track_arr : np.array
        Shape (4, length) where:
        [0] = matches
        [1] = insertions  
        [2] = coverage
        [3] = deletions
    
    Returns:
    --------
    tuple of (positions, match_prop, mismatch_prop, insertion_prop, deletion_prop)
    """
    positions = np.arange(track_arr.shape[1])
    matches = track_arr[0].copy()
    insertions = track_arr[1].copy()
    coverage = track_arr[2].copy()
    deletions = track_arr[3].copy()
    
    # Initialize proportion arrays
    match_prop = np.zeros(len(positions))
    mismatch_prop = np.zeros(len(positions))
    insertion_prop = np.zeros(len(positions))
    deletion_prop = np.zeros(len(positions))
    
    for i in positions:
        total = coverage[i] + insertions[i]
        if total > 0:
            if not np.isnan(matches[i]):
                match_prop[i] = matches[i] / total
                # Mismatches = coverage - matches - deletions
                mismatch_prop[i] = (coverage[i] - matches[i] - deletions[i]) / total
                deletion_prop[i] = deletions[i] / total
            insertion_prop[i] = insertions[i] / total
    
    return positions, match_prop, mismatch_prop, insertion_prop, deletion_prop
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
        
def create_summary_statistics_table(bwa_data, zap_data, model,
                                   out_dir=None, out_prefix=''):
    """
    Create comprehensive summary statistics table per tRNA class.
    """
    sort_order = load_sort_order(model)
    
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


def plot_identity_histogram(bwa_ident_dict, zap_ident_dict, out_pre, out_dir, binwidth=0.025):
    """
    Create identity histograms: separate for each aligner and overlaid.
    """
    # Collect all identities
    bwa_all_idents = []
    for trna, idents in bwa_ident_dict.items():
        bwa_all_idents.extend(idents)
    
    zap_all_idents = []
    for trna, idents in zap_ident_dict.items():
        zap_all_idents.extend(idents)
    
    # Create individual histograms
    fig, ax = plt.subplots(1, figsize=(10, 6))
    sns.histplot(bwa_all_idents, binwidth=binwidth, color='b', alpha=0.75, label='BWA', ax=ax)
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('BWA Alignment Identity Distribution', fontsize=14)
    ax.legend()
    plt.tight_layout()
    out_path_bwa = os.path.join(out_dir, f"{out_pre}_identity_histogram_bwa.pdf")
    fig.savefig(out_path_bwa)
    plt.close(fig)
    
    fig, ax = plt.subplots(1, figsize=(10, 6))
    sns.histplot(zap_all_idents, binwidth=binwidth, color='orange', alpha=0.75, label='Zap', ax=ax)
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('tRNA-zap Alignment Identity Distribution', fontsize=14)
    ax.legend()
    plt.tight_layout()
    out_path_zap = os.path.join(out_dir, f"{out_pre}_identity_histogram_zap.pdf")
    fig.savefig(out_path_zap)
    plt.close(fig)
    
    # Create overlaid histogram
    fig, ax = plt.subplots(1, figsize=(10, 6))
    sns.histplot(bwa_all_idents, binwidth=binwidth, color='b', alpha=0.5, label='BWA', ax=ax)
    sns.histplot(zap_all_idents, binwidth=binwidth, color='orange', alpha=0.5, label='Zap', ax=ax)
    ax.set_xlabel('Identity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Alignment Identity Distribution Comparison', fontsize=14)
    ax.legend()
    plt.tight_layout()
    out_path_both = os.path.join(out_dir, f"{out_pre}_identity_histogram_overlay.pdf")
    fig.savefig(out_path_both)
    plt.close(fig)


def plot_positional_error_barplot(zap_position_ident_dict,
                                   bwa_position_ident_dict,
                                   bwa_ident_dict,
                                   zap_ident_dict,
                                   out_pre,
                                   out_dir,
                                   sort_order):
    """
    Create stacked bar plots showing proportion of match/mismatch/insertion/deletion
    at each position for each tRNA. NOW CORRECTLY USES DELETIONS!
    """
    present_trnas = [trna for trna in sort_order if trna in zap_position_ident_dict]
    
    height_per = 5
    fig, axes = plt.subplots(len(present_trnas), 2, figsize=(16, len(present_trnas) * height_per))
    
    if len(present_trnas) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, trna in enumerate(present_trnas):
        # BWA
        if trna in bwa_position_ident_dict:
            track_arr = bwa_position_ident_dict[trna]
            n_reads = len(bwa_ident_dict[trna]) if trna in bwa_ident_dict else 0
            
            # USE THE HELPER FUNCTION!
            positions, match_prop, mismatch_prop, insertion_prop, deletion_prop = compute_position_error_props(track_arr)
            
            # Stacked bar plot
            axes[idx, 0].bar(positions, match_prop, label='Match', color='green', alpha=0.7, width=1.0)
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
        if trna in zap_position_ident_dict:
            track_arr = zap_position_ident_dict[trna]
            n_reads = len(zap_ident_dict[trna]) if trna in zap_ident_dict else 0
            
            # USE THE HELPER FUNCTION!
            positions, match_prop, mismatch_prop, insertion_prop, deletion_prop = compute_position_error_props(track_arr)
            
            # Stacked bar plot
            axes[idx, 1].bar(positions, match_prop, label='Match', color='green', alpha=0.7, width=1.0)
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
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{out_pre}_positional_error_barplot.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def make_count_matrix(total_read_dict):
    classification_df = pd.DataFrame.from_dict(total_read_dict, orient="index")
    classification_df = classification_df.replace("Failed", "Unmapped")
    all_classes = set(classification_df["zap"]).union(set(classification_df["bwa"]))
    classification_df["bwa"] = pd.Categorical(classification_df["bwa"], categories=all_classes)
    classification_df["zap"] = pd.Categorical(classification_df["zap"], categories=all_classes)
    count_matrix = pd.crosstab(
        classification_df['bwa'],
        classification_df['zap'],
        rownames=['bwa'],
        colnames=['zap'],
        dropna=False
    )
    # Check and drop row if exists
    if "No tRNA" in count_matrix.index:
        count_matrix.drop(["No tRNA"], axis=0, inplace=True)

    # Check and drop column if exists
    if "No tRNA" in count_matrix.columns:
        count_matrix.drop(["No tRNA"], axis=1, inplace=True)
    count_matrix.loc["Unmapped", "Unmapped"] = 0
    count_matrix.index = count_matrix.index.astype(str)
    count_matrix = count_matrix.sort_index(axis=0)
    count_matrix.columns = count_matrix.columns.astype(str)
    count_matrix = count_matrix.sort_index(axis=1)
    return count_matrix


def calculate_per_read_ident(read_ident_dict, id_set):
    zap_ident = [0] * len(id_set)
    bwa_ident = [0] * len(id_set)
    idx = 0
    for key in id_set:
        zap_ident[idx] = read_ident_dict[key]['zap']
        bwa_ident[idx] = read_ident_dict[key]['bwa']
        idx += 1
    df = pd.DataFrame()
    df["zap_ident"] = zap_ident
    df["bwa_ident"] = bwa_ident
    return df


def calculate_misclassified_read_ident(read_ident_dict, mismatch_id_set):
    mismatch_zap_ident = [0] * len(mismatch_id_set)
    mismatch_bwa_ident = [0] * len(mismatch_id_set)
    idx = 0
    for key in mismatch_id_set:
        mismatch_zap_ident[idx] = read_ident_dict[key]['zap']
        mismatch_bwa_ident[idx] = read_ident_dict[key]['bwa']
        idx += 1
    df = pd.DataFrame()
    df["zap_ident"] = mismatch_zap_ident
    df["bwa_ident"] = mismatch_bwa_ident
    return df


def calc_one_aligner_ident(read_ident_dict,
                           bwa_no_zap,
                           zap_no_bwa):
    bwa_no_zap_ident = [0] * len(bwa_no_zap)
    bwa_idx = 0
    for key in bwa_no_zap:
        bwa_no_zap_ident[bwa_idx] = read_ident_dict[key]['bwa']
        bwa_idx += 1

    zap_no_bwa_ident = [0] * len(zap_no_bwa)
    zap_idx = 0
    for key in zap_no_bwa:
        zap_no_bwa_ident[zap_idx] = read_ident_dict[key]['zap']
        zap_idx += 1

    return bwa_no_zap_ident, zap_no_bwa_ident


def make_id_sets(total_read_dict):
    id_set = set()
    mismatch_id_set = set()
    bwa_no_zap = set()
    zap_no_bwa = set()
    for read_id in total_read_dict:
        zap = total_read_dict[read_id]['zap']
        bwa = total_read_dict[read_id]['bwa']
        if zap != 'Unmapped' and zap != 'No tRNA' and zap != 'Failed' and bwa != 'Unmapped' and bwa != 'No tRNA' and bwa != 'Failed':
            id_set.add(read_id)
            if zap != bwa:
                mismatch_id_set.add(read_id)
        elif bwa != 'Unmapped' and bwa != 'No tRNA' and bwa != 'Failed':
            bwa_no_zap.add(read_id)
        elif zap != 'Unmapped' and zap != 'No tRNA' and zap != 'Failed':
            zap_no_bwa.add(read_id)

    return id_set, mismatch_id_set, bwa_no_zap, zap_no_bwa    

def get_zap_model(zap_ref):
    for seq in pysam.FastxFile(zap_ref):
        first_ref = seq.name
        break
    if 'esch' in first_ref:
        return 'e_coli'
    if 'sac' in first_ref:
        return 'yeast'
    else:
        return 'human'
        
def print_alignment_summary(count_df):
    """
    Print summary of total aligned reads for BWA vs Zap.
    """
    # Exclude failed/unmapped reads
    exclude_categories = ['Unmapped', 'Failed', 'No trna']
    aligned_df = count_df[~count_df['trna'].isin(exclude_categories)]
    
    # Sum up aligned reads per aligner
    bwa_total = aligned_df[aligned_df['aligner'] == 'bwa']['count'].sum()
    zap_total = aligned_df[aligned_df['aligner'] == 'zap']['count'].sum()
    
    # Calculate percent change (BWA as baseline = 100%)
    if bwa_total > 0:
        percent_change = ((zap_total - bwa_total) / bwa_total) * 100
        zap_as_percent_of_bwa = (zap_total / bwa_total) * 100
    else:
        percent_change = 0
        zap_as_percent_of_bwa = 0
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    print(f"BWA aligned reads:  {bwa_total:>12,} (100.0%)")
    print(f"Zap aligned reads:  {zap_total:>12,} ({zap_as_percent_of_bwa:.1f}%)")
    print(f"Difference:         {zap_total - bwa_total:>+12,} ({percent_change:+.1f}%)")
    print("="*60 + "\n")

def safe_divide_for_plot(ident, coverage, insertions):
    """
    Safely divide identity by coverage+insertions, handling zero denominators.
    """
    denominator = coverage + insertions
    with np.errstate(divide='ignore', invalid='ignore'):
        result = ident / denominator
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def create_figures(
        bwa_ref,
        zap_ref,
        bwa_bam,
        zap_bam,
        threads,
        out_pre,
        out_dir,
        per_class_identity=True,
        class_counts=True,
        plot_deltas=True,
        position_ident=True,
        alignment_heatmap=True,
        per_read_ident=True,
        misclassified_ident=True,
        one_aligner_ident=True,
        classification_heatmap=True,
        hexbin_delta=True,
        hexbin_gridsize=30,
        summary_table=True,          
        identity_histogram=True,     
        positional_error_bars=True):
    
    suppress_plotting_warnings()
    ref_convert, bwa_mem_ref_dict, zap_ref_dict, bwa_ref_lens, zap_ref_lens = conversion_dicts(bwa_ref, zap_ref)
    # Calculate and unpack all the relevant datatypes
    (ref_set,
     total_read_dict,
     bwa_ident_dict,
     zap_ident_dict,
     bwa_position_ident_dict,
     zap_position_ident_dict,
     read_ident_dict,
     exclusion_read_id_set) = per_read_output(bwa_bam,
                                              zap_bam,
                                              bwa_mem_ref_dict,
                                              bwa_ref_lens,
                                              zap_ref_dict,
                                              zap_ref_lens,
                                              threads)
    df = pd.DataFrame([
                          {'trna': key, 'ident': value, 'aligner': 'bwa'}
                          for key, values in bwa_ident_dict.items()
                          for value in values
                      ] + [
                          {'trna': key, 'ident': value, 'aligner': 'zap'}
                          for key, values in zap_ident_dict.items()
                          for value in values
                      ])

    #with open("total_read_dict.pkl", "wb") as outfile:
    #    pickle.dump(total_read_dict, outfile)

    zap_model = get_zap_model(zap_ref)

    if zap_model == 'e_coli':
        viz_path = files('trnazap').joinpath('visualize')
        with open(str(viz_path / 'alignment_viz' / 'ecoli_label_sort_order.json'), 'r') as infile:
            sort_order = json.load(infile)

    elif zap_model == 'yeast':
        viz_path = files('trnazap').joinpath('visualize')
        with open(str(viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'), 'r') as infile:
            sort_order = json.load(infile)

    else: 
        viz_path = files('trnazap').joinpath('visualize') 
        with open(str(viz_path / 'alignment_viz' / 'yeast_label_sort_order.json'), 'r') as infile:
            sort_order = json.load(infile)

    if per_class_identity:
        per_class_identity_plot(df, 
                                out_pre, 
                                out_dir, 
                                sort_order)

    alignment_count_df = alignment_counts(total_read_dict)
    print_alignment_summary(alignment_count_df)

    if class_counts:
        class_count_plots(alignment_count_df, 
                          out_pre, 
                          out_dir, 
                          sort_order)

    if plot_deltas:
        deltas, delta_label = count_deltas(alignment_count_df)
        class_delta_plots(deltas,
                          delta_label,
                          out_pre,
                          out_dir, 
                          sort_order)

    if position_ident:
        position_ident_plot(zap_position_ident_dict,
                            bwa_position_ident_dict,
                            out_pre,
                            out_dir,
                            sort_order
                            )

    count_matrix = make_count_matrix(total_read_dict)

    if alignment_heatmap:
        plot_alignment_heatmap(count_matrix,
                               out_pre,
                               out_dir,
                               sort_order
                              )

    id_set, mismatch_id_set, bwa_no_zap, zap_no_bwa = make_id_sets(total_read_dict)

    run_statistical_comparisons(read_ident_dict, id_set, mismatch_id_set, out_pre, out_dir)

    if hexbin_delta:
        plot_hexbin_delta(read_ident_dict, id_set, out_pre, out_dir, 
                         gridsize=hexbin_gridsize, save_data=True)
    
    if per_read_ident:
        per_read_df = calculate_per_read_ident(read_ident_dict, id_set)
        plot_per_read_ident(per_read_df, out_pre, out_dir)

    if misclassified_ident:
        misclassified_per_read_df = calculate_misclassified_read_ident(read_ident_dict,
                                                                       mismatch_id_set)
        plot_per_read_miss_classified_ident(misclassified_per_read_df, out_pre, out_dir)

    if one_aligner_ident:
        bwa_no_zap_ident, zap_no_bwa_ident = calc_one_aligner_ident(read_ident_dict,
                                                                    bwa_no_zap,
                                                                    zap_no_bwa)
        plot_one_aligner_hist_plots(bwa_no_zap_ident, zap_no_bwa_ident, out_pre, out_dir)

    if summary_table:
        summary_df = create_summary_statistics_table(
            bwa_ident_dict,
            zap_ident_dict,
            bwa_position_ident_dict,
            zap_position_ident_dict,
            read_ident_dict,
            total_read_dict,
            out_pre,
            out_dir,
            sort_order
        )
    if identity_histogram:
        plot_identity_histogram(bwa_ident_dict, zap_ident_dict, out_pre, out_dir, binwidth=0.025)

    t0 = time()
    print(f"{t0=}")
    if positional_error_bars:
        plot_positional_error_barplot(
            zap_position_ident_dict,
            bwa_position_ident_dict,
            bwa_ident_dict,
            zap_ident_dict,
            out_pre,
            out_dir,
            sort_order
        )
    print(f"Positional Error Bars took: {(time()-t0)/60} minutes")
