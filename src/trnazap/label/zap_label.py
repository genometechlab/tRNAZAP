import pysam
import numpy as np
from numba import njit, jit
import pickle
from tqdm import tqdm
import argparse
from collections import defaultdict


@njit
def annot_from_read(ref_pos, ref_len, tRNA_code, mv_table, ts, ref_start, ref_end, fragment):
    n = len(ref_pos)
    x_label = np.full(n, -1, dtype=np.int32)
    seen_ref_start = False
    past_ref_end = False

    prev_pos = None
    for i in range(n):
        pos = ref_pos[i]
        
        # Update flags
        if pos != -1:
            if pos == ref_start:
                seen_ref_start = True
            if pos >= ref_end - 1:  # ref_end is exclusive in pysam
                past_ref_end = True
        
        # Assign labels
        if pos == -1:  # Unmapped position
            if not seen_ref_start:
                # Haven't seen ref_start yet - 5' softclip
                x_label[i] = 2
            elif past_ref_end:
                # Past ref_end - 3' softclip
                x_label[i] = 1
            elif prev_pos < 36:
                x_label[i] = 2
            elif prev_pos > ref_len - 41:
                # Past ref_end - 3' softclip
                x_label[i] = 1
            elif seen_ref_start and not past_ref_end and prev_pos <= ref_len - 41:
                # Between ref_start and ref_end - internal deletion
                x_label[i] = tRNA_code
            else:
                assert 1 == 2, f"{ref_pos} : {i} : {ref_pos[i]}"
        else:  # Mapped position
            if pos < 36:
                # 5' adapter region
                x_label[i] = 2
            elif past_ref_end or pos > ref_len - 41:
                # 3' adapter region
                x_label[i] = 1
            else:
                # tRNA region
                x_label[i] = tRNA_code
            prev_pos = pos
    
    mv_index = np.where(mv_table == 1)[0] - 1
    stride_size = mv_table[0]
    annotations = np.full(shape=(ts + (len(mv_table) - 1) * stride_size), fill_value = -1)
    annotations[:ts] = 0
    flip_label = np.flip(x_label)
    
    for i in range(len(flip_label)):
        if i + 1 >= len(flip_label):
            annotations[ts + mv_index[i] * stride_size:] = flip_label[i]
        else:
            annotations[ts + mv_index[i] * stride_size:ts + mv_index[i+1] * stride_size] = flip_label[i]
    
    rle = run_length_encode_annotations(annotations)
    
    return rle, fragment

@njit
def run_length_encode_annotations(annotations):
    current = annotations[0]
    count = 1
    encoded = []
    for i in range(1, annotations.shape[0]):
        if annotations[i] != current:
            encoded.append((current, count))
            count = 1
            current = annotations[i]
        else:
            count += 1
    
    encoded.append((current, count))
    #print(encoded)
    return encoded

def check_identity(read, ref_seq, ref_start, ref_end):
    matches = 0
    mismatches = 0
    deletions = 0
    insertions = 0
    seen_ref_start = False
    past_ref_end = False
    prev_pos = None
    ref_len = len(ref_seq)

    for pair in read.get_aligned_pairs():
        pos = pair[1]
        read_pos = pair[0]
            
        # Update flags
        if pos is not None:
            if pos == ref_start:
                seen_ref_start = True
            if pos >= ref_end - 1:  # ref_end is exclusive in pysam
                past_ref_end = True
        
        # Assign labels
        if pos is None:  # Unmapped position
            if not seen_ref_start:
                continue
            elif past_ref_end:
                # Past ref_end - 3' softclip
                continue
            elif prev_pos < 36:
                continue
            elif prev_pos > ref_len - 41:
                # Past ref_end - 3' softclip
                continue
            elif seen_ref_start and not past_ref_end and prev_pos <= ref_len - 41:
                # Between ref_start and ref_end - internal deletion
                insertions += 1
            else:
                assert 1 == 2, f"{ref_pos} : {i} : {ref_pos[i]}"
        else:  # Mapped position
            if pos < 36:
                # 5' adapter region
                prev_pos = pos
                continue
            elif past_ref_end or pos > ref_len - 41:
                # 3' adapter region
                prev_pos = pos
                continue
            elif read_pos is None:
                deletions += 1
            else:
                # tRNA region
                if read.query_sequence[read_pos] == ref_seq[pos]:
                    matches += 1
                else:
                    mismatches += 1
            prev_pos = pos
            
    return matches, mismatches, insertions, deletions
            
@njit
def edit_dist(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[m][n]

def disambiguate(read, tRNA_class_entry):
    five_p_offset = 36
    if 'edit_dist' in tRNA_class_entry:

        encoded_query_seq = np.array([ord(x) for x in read.query_sequence])
        seq_1_1 = tRNA_class_entry['seq_1-1']
        seq_2_1 = tRNA_class_entry['seq_2-1']
        seq_1_1_dist = edit_dist(encoded_query_seq, seq_1_1)
        seq_2_1_dist = edit_dist(encoded_query_seq, seq_2_1)

        if seq_1_1_dist == seq_2_1_dist:
            return None
        elif seq_1_1_dist < seq_2_1_dist:
            return '1-1'
        else:
            return '2-1'

    else:
        key_count = {key:len(tRNA_class_entry[key]) for key in tRNA_class_entry}
        example_key = list(key_count.keys())[0]
        for pair in read.get_aligned_pairs():
            if pair[1] is not None and pair[1] in tRNA_class_entry[example_key] and pair[0] is not None:
                seq_base = read.query_sequence[pair[0]]
                for key in list(key_count.keys()):
                    if tRNA_class_entry[key][pair[1]] == seq_base:
                        key_count[key] -= 1
                        
        no_match = True
        match_key = None
        for key, value in key_count.items():
            if value == 0:
                assert no_match is True
                no_match = False
                match_key = key

        if no_match:
            return None

        else:
            return match_key    

def zap_label(bam, ref, out, decoder_dict):
    ref_lens = {}
    ref_seqs = {}
    tRNA_labels = {}
    tRNA_base_name = None
    count_dict = defaultdict(int)
    fxf = pysam.FastxFile(ref) #fxf needs to be non-subset version
    af = pysam.AlignmentFile(bam)
    for i, tRNA in enumerate(fxf):            
        ref_lens[tRNA.name] = len(tRNA.sequence)
        ref_seqs[tRNA.name] = tRNA.sequence
        tRNA_labels[tRNA.name] = i + 3

    if decoder_dict is not None:
        with open(decoder_dict, 'rb') as infile:
            decoder_dict = pickle.load(infile)

    out_dict = {}
    for read in tqdm(af.fetch()):

            
        if read.is_unmapped or read.mapping_quality == 0 or read.is_secondary or read.is_supplementary or read.has_tag('pi') or read.get_tag('ns') >= 1000000:
            continue

        if abs(max(read.reference_start, 36) - min(read.reference_end, ref_lens[read.reference_name]-41)) < 15:
            continue
        
        ref_positions = np.array(read.get_reference_positions(full_length=True))
        
        ref_positions[ref_positions == None] = -1
        ref_positions = np.array(ref_positions, dtype=np.int32)

        fragment = False
        if read.reference_start > 36 or read.reference_end < ref_lens[read.reference_name]-41:
            fragment = True

        matches, mismatches, insertions, deletions = check_identity(read, ref_seqs[read.reference_name], read.reference_start, read.reference_end)

        if matches / (matches + mismatches + insertions + deletions) < 0.75:
            continue

        ref_name_tmp = read.reference_name

        if decoder_dict is not None:
            if 'mito' not in read.reference_name:
                split_ref = read.reference_name.split('_')[-1].split('-')
                assert len(split_ref) == 5
                encoder = f"{split_ref[1]}-{split_ref[2]}"
                decoder = f"{split_ref[3]}-{split_ref[4]}"
                if encoder in decoder_dict:
                    dis_amb_result = disambiguate(read, decoder_dict[encoder])
                    if dis_amb_result is None:
                        continue
                    ref_name_tmp = '-'.join(ref_name_tmp.split('-')[:-2]) + '-' + dis_amb_result
        
        count_dict[ref_name_tmp] += 1
        out_dict[read.query_name] = annot_from_read(ref_positions, 
                                                    ref_lens[read.reference_name],
                                                    tRNA_labels[ref_name_tmp], 
                                                    np.array(read.get_tag('mv'), dtype=int), 
                                                    read.get_tag('ts'), 
                                                    read.reference_start, 
                                                    read.reference_end, 
                                                    fragment)
        
    
    with open(out, "wb") as outfile:
        pickle.dump((tRNA_labels, out_dict), outfile)

    for key, value in count_dict.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", required=True, help="Aligned tRNA file")
    parser.add_argument("--ref", required=True, help="Reference (should be long splints)")
    parser.add_argument("--out", required=True, help="Outpath")
    parser.add_argument("--decoder_dict", required=False, default = None, help="Decoder disambiguation dict, if this is not provided it is assumed all reads belong to their primary aligned class")

    args = parser.parse_known_args()[0]
    zap_label(args.bam,
              args.ref,
              args.out,
              args.decoder_dict
              )