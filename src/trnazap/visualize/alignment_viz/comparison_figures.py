from .read_pass_for_classification import ref_conversion, multiprocess_trna_data, merge_multiprocess
import pysam
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing
from matplotlib.colors import LogNorm
import pickle

plt.rcParams['pdf.fonttype'] = 42
plt.switch_backend('agg')
sns.set(rc={'figure.figsize': (11.7, 8.27)})


def per_class_identity_plot(in_df, out_pre, out_dir):
    color_dict = {
        'bwa': 'b',
        'zap': 'orange'
    }
    fig, ax = plt.subplots(1)
    fig.set_figheight(2)
    fig.set_figwidth(8)
    sns.boxenplot(data=in_df, x='trna', y='ident', hue='aligner', palette=color_dict)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    out_path = os.path.join(out_dir, f"{out_pre}_per_class_identity.pdf")
    fig.get_figure().savefig(out_path)


def class_count_plots(count_df, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    fig.set_figheight(10)
    fig.set_figwidth(32)
    sns.barplot(data=count_df, x='trna', y='count', hue='aligner')
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yscale('log')
    out_path = os.path.join(out_dir, f"{out_pre}_class_counts.pdf")
    fig.get_figure().savefig(out_path)


def class_delta_plots(deltas, delta_label, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    fig.set_figwidth(14)
    sns.barplot(y=deltas, x=[i for i in range(len(deltas))])
    plt.xticks([i for i in range(len(deltas))], delta_label, rotation=90)
    out_path = os.path.join(out_dir, f"{out_pre}_deltas_class_counts.pdf")
    fig.get_figure().savefig(out_path)


def position_ident_plot(zap_position_ident_dict,
                        bwa_position_ident_dict,
                        out_pre,
                        out_dir):
    
    height_per = 5
    fig, ax = plt.subplots(len(zap_position_ident_dict), 1)
    fig.set_figheight(max(len(bwa_position_ident_dict),
                          len(zap_position_ident_dict)) * height_per)
    t_to_idx = {t: i for i, t in enumerate(zap_position_ident_dict.keys())}
    for dict_tuple in zap_position_ident_dict.items():
        trna, (ident, insertions, coverage) = dict_tuple
        i = t_to_idx[trna]
        ax[i].set_title(trna)
        coverage = coverage
        ident = ident / (coverage + insertions)
        # coverage = coverage / max(coverage)
        sns.lineplot(ident, ax=ax[i], color='orange', drawstyle='steps-mid', label="zap", legend=False)

    for dict_tuple in bwa_position_ident_dict.items():
        trna, (ident, insertions, coverage) = dict_tuple
        if trna not in t_to_idx:
            continue
        i = t_to_idx[trna]
        coverage = coverage
        ident = ident / (coverage + insertions)
        # coverage = coverage / max(coverage)
        sns.lineplot(ident, ax=ax[i], color='blue', drawstyle='steps-mid', linestyle="--", label="bwa", legend=False)
    ax[0].legend(loc="best")
    out_path = os.path.join(out_dir, f"{out_pre}_positional_identity.pdf")
    fig.get_figure().savefig(out_path)


def plot_alignment_heatmap(count_matrix,
                           out_pre,
                           out_dir):
    fig, ax = plt.subplots(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    sns.heatmap(count_matrix, norm=LogNorm(), cmap='viridis', cbar=True)
    out_path = os.path.join(out_dir, f"{out_pre}_alignment_classification_heatmap.pdf")
    fig.get_figure().savefig(out_path)


def plot_classification_heatmap(classification_df,
                                out_pre,
                                out_dir):

    fig, ax = plt.subplots(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    sns.heatmap(classification_df, norm=LogNorm(), cmap='viridis', cbar=True)
    out_path = os.path.join(out_dir, f"{out_pre}_classification_only_heatmap.pdf")
    fig.get_figure().savefig(out_path)


def plot_one_aligner_hist_plots(bwa_no_zap_ident, zap_no_bwa_ident, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    print(f"{len(bwa_no_zap_ident)=}")
    print(f"{len(zap_no_bwa_ident)=}")
    if len(bwa_no_zap_ident) > 1:
        sns.histplot(bwa_no_zap_ident, binwidth=0.025, label="bwa", color='b', alpha=0.75)
    if len(zap_no_bwa_ident) > 1:
        sns.histplot(zap_no_bwa_ident, binwidth=0.025, label="zap", color='orange', alpha=0.75)
    plt.legend(loc='best')
    out_path = os.path.join(out_dir, f"{out_pre}_one_aligner_only_ident_hist_plot.pdf")
    fig.get_figure().savefig(out_path)


def plot_per_read_miss_classified_ident(misclassified_ident_df, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    sns.histplot(data=misclassified_ident_df, x='zap_ident', y='bwa_ident', binwidth=0.025)
    out_path = os.path.join(out_dir, f"{out_pre}_misclassified_read_ident.pdf")
    fig.get_figure().savefig(out_path)


def plot_per_read_ident(read_ident_df, out_pre, out_dir):
    fig, ax = plt.subplots(1)
    fig.set_figwidth(8)
    fig.set_figheight(8)
    sns.histplot(data=read_ident_df, x='zap_ident', y='bwa_ident', binwidth=0.0125)
    out_path = os.path.join(out_dir, f"{out_pre}_pre_read_ident.pdf")
    fig.get_figure().savefig(out_path)


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

    bwa_mem_ref_dict = {}
    zap_ref_dict = {}

    for seq in pysam.FastxFile(bwa_ref):
        bwa_mem_ref_dict[seq.name] = seq.sequence
    for seq in pysam.FastxFile(zap_ref):
        zap_ref_dict[seq.name] = seq.sequence#[36:-42]
    ref_convert = ref_conversion(bwa_ref,
                                 zap_ref)

    bwa_ref_lens = {}
    for key in bwa_mem_ref_dict:
        bwa_ref_lens[key] = len(bwa_mem_ref_dict[key])
    zap_ref_lens = {}
    for key in zap_ref_dict:
        zap_ref_lens[key] = len(zap_ref_dict[key])

    for key, value in zap_ref_dict.items():
        print(f"{key} : {value}")
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
                                {'trna': key.split("_")[-1][5:], 'count': value, 'aligner': 'bwa'}
                                for key, value in bwa_alignments.items()]
                            + [
                                {'trna': key.split("_")[-1][5:], 'count': value, 'aligner': 'zap'}
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
        classification_heatmap=True
):
    print(zap_ref)
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
                          {'trna': key.split("_")[-1][5:], 'ident': value, 'aligner': 'bwa'}
                          for key, values in bwa_ident_dict.items()
                          for value in values
                      ] + [
                          {'trna': key.split("_")[-1][5:], 'ident': value, 'aligner': 'zap'}
                          for key, values in zap_ident_dict.items()
                          for value in values
                      ])

    with open("total_read_dict.pkl", "wb") as outfile:
        pickle.dump(total_read_dict, outfile)

    if per_class_identity:
        per_class_identity_plot(df, out_pre, out_dir)

    alignment_count_df = alignment_counts(total_read_dict)

    if class_counts:
        class_count_plots(alignment_count_df, out_pre, out_dir)

    if plot_deltas:
        deltas, delta_label = count_deltas(alignment_count_df)
        class_delta_plots(deltas,
                          delta_label,
                          out_pre,
                          out_dir)

    if position_ident:
        position_ident_plot(zap_position_ident_dict,
                            bwa_position_ident_dict,
                            out_pre,
                            out_dir
                            )

    count_matrix = make_count_matrix(total_read_dict)

    if alignment_heatmap:
        plot_alignment_heatmap(count_matrix,
                               out_pre,
                               out_dir)

    id_set, mismatch_id_set, bwa_no_zap, zap_no_bwa = make_id_sets(total_read_dict)

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
'''
    if classification_heatmap:
        classification_df = make_classification_df(bwa_bam, zap_bam, exclusion_read_id_set)
        plot_classification_heatmap(classification_df, out_pre, out_dir)
'''
