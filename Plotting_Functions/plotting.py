"""Module for plotting results of tRNA alignments."""

import numpy as np


def visualize_seqs(sub_sequence, ref_sequence, cigar_tuples, omit_softclip=True):
    """Creaste ASCII representations of alignments."""
    # Initialized
    out_ref_string = ""
    operation_string = ""
    out_seq_string = ""
    ref_index = 0
    seq_index = 0
    for tup in cigar_tuples:
        action = tup[0]
        count = tup[1]
        if action == 7:
            out_ref_string = (
                out_ref_string + ref_sequence[ref_index : ref_index + count]
            )
            out_seq_string = (
                out_seq_string + sub_sequence[seq_index : seq_index + count]
            )
            operation_string = operation_string + "|" * count
            ref_index += count
            seq_index += count
        if action == 4 and omit_softclip:
            continue

        # Note if the subsequence is passed this will not work
        if action == 4 and not omit_softclip:
            out_ref_string = out_ref_string + " " * count
            out_seq_string = (
                out_seq_string + sub_sequence[seq_index : seq_index + count]
            )
            operation_string = operation_string + " " * count
            seq_index += count

        if action == 8:
            out_ref_string = (
                out_ref_string + ref_sequence[ref_index : ref_index + count]
            )
            out_seq_string = (
                out_seq_string + sub_sequence[seq_index : seq_index + count]
            )
            operation_string = operation_string + "*" * count
            ref_index += count
            seq_index += count

        if action == 1:
            out_ref_string = out_ref_string + "-" * count
            out_seq_string = (
                out_seq_string + sub_sequence[seq_index : seq_index + count]
            )
            operation_string = operation_string + "." * count
            seq_index += count

        if action == 2:
            out_ref_string = (
                out_ref_string + ref_sequence[ref_index : ref_index + count]
            )
            out_seq_string = out_seq_string + "-" * count
            operation_string = operation_string + "." * count
            ref_index += count

    array = np.array(
        [list(out_ref_string), list(operation_string), list(out_seq_string)]
    )

    print(array)
