import pysam
import numpy as np
from numba import njit
import pickle
from tqdm import tqdm
from collections import defaultdict

FIVE_PRIME_ADAPTER_LEN = 36   # 5' ligation oligo length
THREE_PRIME_ADAPTER_LEN = 41  # 3' ligation oligo length
MIN_TRNA_COVERAGE = 15        # minimum tRNA region coverage to process a read


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
            elif prev_pos < FIVE_PRIME_ADAPTER_LEN:
                x_label[i] = 2
            elif prev_pos > ref_len - THREE_PRIME_ADAPTER_LEN:
                # Past ref_end - 3' softclip
                x_label[i] = 1
            elif seen_ref_start and not past_ref_end and prev_pos <= ref_len - THREE_PRIME_ADAPTER_LEN:
                # Between ref_start and ref_end - internal deletion
                x_label[i] = tRNA_code
            else:
                assert 1 == 2, f"{ref_pos} : {i} : {ref_pos[i]}"
        else:  # Mapped position
            if pos < FIVE_PRIME_ADAPTER_LEN:
                # 5' adapter region
                x_label[i] = 2
            elif past_ref_end or pos > ref_len - THREE_PRIME_ADAPTER_LEN:
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
            elif prev_pos < FIVE_PRIME_ADAPTER_LEN:
                continue
            elif prev_pos > ref_len - THREE_PRIME_ADAPTER_LEN:
                # Past ref_end - 3' softclip
                continue
            elif seen_ref_start and not past_ref_end and prev_pos <= ref_len - THREE_PRIME_ADAPTER_LEN:
                # Between ref_start and ref_end - internal deletion
                insertions += 1
        else:  # Mapped position
            if pos < FIVE_PRIME_ADAPTER_LEN:
                # 5' adapter region
                prev_pos = pos
                continue
            elif past_ref_end or pos > ref_len - THREE_PRIME_ADAPTER_LEN:
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

def disambiguate_human_ivt(read, specific_tRNA_entry):
    #specific tRNA entry is reference position matched w/ expected base, if any fail reject read
    match_count = 0
    required_matches = len(specific_tRNA_entry)
    for pair in read.get_aligned_pairs():
        if pair[1] is not None and pair[1] in specific_tRNA_entry:
            if pair[0] is None: #Deletion at required ref_pos
                return None
            if read.query_sequence[pair[0]] != specific_tRNA_entry[pair[1]]: #Mismatch
                return None
            match_count += 1

    assert match_count <= required_matches
    if match_count != required_matches:
        return None
    return read.reference_name

def disambiguate(read, tRNA_class_entry):

    if 'edit_dist' in tRNA_class_entry:

        encoded_query_seq = np.array([ord(x) for x in read.query_sequence])
        seq_1_1 = tRNA_class_entry['seq_1-1']
        seq_2_1 = tRNA_class_entry['seq_2-1']
        seq_1_1_dist = edit_dist(encoded_query_seq, seq_1_1) / max(len(encoded_query_seq), len(seq_1_1))
        seq_2_1_dist = edit_dist(encoded_query_seq, seq_2_1) / max(len(encoded_query_seq), len(seq_2_1))

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

def zap_label(bam, ref, out, decoder_dict, min_ident, human_ivt = False):
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
        print(f"{tRNA.name} {ref_lens[tRNA.name]}")

    if decoder_dict is not None:
        with open(decoder_dict, 'rb') as infile:
            decoder_dict = pickle.load(infile)

    out_dict = {}
    for read in tqdm(af.fetch()):
        
        if read.is_unmapped or read.mapping_quality == 0 or read.is_secondary or read.is_supplementary or read.has_tag('pi') or read.get_tag('ns') >= 1000000:
            continue

        if abs(max(read.reference_start, FIVE_PRIME_ADAPTER_LEN) - min(read.reference_end, ref_lens[read.reference_name] - THREE_PRIME_ADAPTER_LEN)) < MIN_TRNA_COVERAGE:
            continue

        ref_positions = np.array(read.get_reference_positions(full_length=True))
        
        ref_positions[ref_positions == None] = -1
        ref_positions = np.array(ref_positions, dtype=np.int32)

        fragment = False
        if read.reference_start > FIVE_PRIME_ADAPTER_LEN or read.reference_end < ref_lens[read.reference_name] - THREE_PRIME_ADAPTER_LEN:
            fragment = True

        matches, mismatches, insertions, deletions = check_identity(read, ref_seqs[read.reference_name], read.reference_start, read.reference_end)
        
        if matches / (matches + mismatches + insertions + deletions) < min_ident:
            continue

        ref_name_tmp = read.reference_name

        if decoder_dict is not None:
            gln_ctg = False
            if 'His-GTG' in read.reference_name or 'Ile-GAT' in read.reference_name or 'SeC-TCA' in read.reference_name or 'Tyr-ATA' in read.reference_name or "Leu-CAA-5-1" in read.reference_name:
                pass
            elif 'mito' not in read.reference_name and 'Mt_tRNA' not in read.reference_name:
                split_ref = read.reference_name.split('_')[-1].split('-')
                assert len(split_ref) == 5 ,f"{read.reference_name}"
                encoder = f"{split_ref[1]}-{split_ref[2]}"
                decoder = f"{split_ref[3]}-{split_ref[4]}"
                if human_ivt:
                    targets = decoder_dict[encoder][read.reference_name]

                    dis_amb_result = disambiguate_human_ivt(read, targets)

                    if dis_amb_result is None:
                        continue
                    ref_name_tmp = read.reference_name
                    
                else:
                    if encoder == "Gln-CTG":
                        gln_ctg = True
                        encoder = "Gln-TTG"
                    if encoder in decoder_dict:
                        dis_amb_result = disambiguate(read, decoder_dict[encoder])
                        if dis_amb_result is None:
                            continue
                        ref_name_tmp = '-'.join(ref_name_tmp.split('-')[:-2]) + '-' + dis_amb_result
                        if dis_amb_result == "Gln-CTG-1-1":
                            ref_name_tmp = '-'.join([ref_name_tmp.split('-')[0], dis_amb_result])
                        if gln_ctg and dis_amb_result in set(["2-1", "1-1", "3-1"]):
                            encoder = "Gln-TTG"
                            ref_name_tmp = '-'.join([ref_name_tmp.split('-')[0], encoder, dis_amb_result])
        
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
