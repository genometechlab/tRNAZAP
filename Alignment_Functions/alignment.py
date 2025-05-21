"""Module that executes alignement functions.

This module has utility functions that can be applied
broadly to situtation that mimic the tRNA set up: Ionic
current based delineation of sequence encapsulating the
reference region.
"""

import numpy as np
import pysam
from numba import njit


@njit
def wagner_fisher(s: str, t: str):
    """Optimized Levenshtein distance calculation for tRNA.

    This implementation is specialized for tRNA alignment
    with modified scoring rules
    that allow free end gaps in the reference sequence (t).
    This accommodates partial
    alignments where only a portion of the query may match
    the reference.

    Args:
        s: Query sequence (the observed tRNA sequence based
        on model prediction)
        t: Reference sequence (the known tRNA template of
        the predicted class)

    Returns:
        tuple: A tuple containing:
            - float: The edit distance score
            - ndarray: The complete dynamic programming
            matrix
    """
    m, n = len(s), len(t)
    d = np.zeros(shape=(m + 1, n + 1), dtype=np.float64)

    # Initialize first row with zeros to allow free start gaps in reference
    # This means the query can start aligning at any position in the reference
    for j in range(n + 1):
        d[0, j] = j

    # Initialize first column with increasing values
    # This represents the cost of deleting characters from the query
    for i in range(1, m + 1):
        d[i, 0] = 0

    # Fill the matrix
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            d[i, j] = min(
                d[i - 1, j] + 1,  # deletion from query
                d[i, j - 1] + 1,  # insertion into query
                d[i - 1, j - 1] + substitution_cost,
            )  # substitution/match

    return d[m, n], d


def calcuate_vertical_traversal(d):
    """
    Calculate the vertical traversal path.

    For each position in the last column of the dynamic
    programming matrix, this function traces back to find
    where the alignment would start in the reference
    sequence. This helps identify the optimal subset of
    the query that best matches the reference.

    Args:
        d: Dynamic programming matrix from wagner_fisher
        algorithm

    Returns:
        tuple: A tuple containing:
            - ndarray: Matrix mapping each position to its
            starting point
            - ndarray: Array of vertical traversal distances
            for each position
    """
    # Used AI to comment code below. Should be duplicate
    """
    # Keep track of where each position started
    trav_mat = np.full((d.shape[0], d.shape[1]), -1)

    # Array for tracking the vertical traversal distance
    traversal_distance = np.full((d.shape[0]), d.shape[0]
    + d.shape[1] + 1)

    #Iterate through each vertical start location in the
    #last column (possible alignment end points)
    #and calculate what the start position was. Use
    the trav_mat to short circuit pre-calculated values
    for seed in list(range(d.shape[0]))[::-1]:
        tmp_x, tmp_y = (seed, d.shape[1] - 1)
        positions = [0] * (tmp_x + tmp_y)
        positions[0] = (seed, -1)
        position_tracker = 1
        while tmp_y > 1:
            deletion_score = d[tmp_x - 1, tmp_y] if
            tmp_x >= 1 else np.inf
            insertion_score = d[tmp_x, tmp_y - 1] if
            tmp_y >= 1 else np.inf
            sub_or_match_score = d[tmp_x - 1, tmp_y - 1]
            if tmp_x >= 1 and tmp_y >= 1 else np.inf

            if sub_or_match_score <= deletion_score and
            sub_or_match_score <= insertion_score:
                if trav_mat[tmp_x -1][tmp_y - 1] != -1:
                    start = trav_mat[tmp_x - 1][tmp_y - 1]
                    tmp_y = 1
                else:
                    positions[position_tracker] = (
                    tmp_x - 1, tmp_y - 1)  # Substitution
                    tmp_x -= 1
                    tmp_y -= 1
                    start = tmp_x
                    position_tracker += 1

            elif insertion_score <= deletion_score:
                # In terms of transforming query to
                #reference:
                # Insertion into  query
                if trav_mat[tmp_x][tmp_y - 1] != -1:
                    start = trav_mat[tmp_x][tmp_y - 1]
                    tmp_y = 1
                else:
                    positions[position_tracker] = (tmp_x,
                    tmp_y - 1)  # Deletion
                    tmp_y -= 1
                    start = tmp_x
                    position_tracker += 1

            else:
                # Deletion from query
                if trav_mat[tmp_x - 1][tmp_y] != -1:
                    start = trav_mat[tmp_x - 1][tmp_y]
                    tmp_y = 1
                else:
                    positions[position_tracker] = (tmp_x -
                    1, tmp_y)  # Insertion
                    tmp_x -= 1
                    start = tmp_x
                    position_tracker += 1

        for p in range(position_tracker):
            trav_mat[positions[p]] = start

        traversal_distance[seed] = abs(seed -
        trav_mat[(seed, -1)])

    return trav_mat, traversal_distance
    """
    # Initialization matrix to store the start position for each cell in traversal
    trav_mat = np.full((d.shape[0], d.shape[1]), -1)

    # For each position in the query, store the vertical distance traversed
    # (initialized to a high value - worst case would be traversing the entire matrix)
    traversal_distance = np.full((d.shape[0]), d.shape[0] + d.shape[1] + 1)

    # Iterate through potential alignment end positions
    # (last column), starting from bottom
    # We work backward ([::-1]) to prioritize longer
    # alignments
    for seed in list(range(d.shape[0]))[::-1]:
        tmp_x, tmp_y = (seed, d.shape[1] - 1)  # Start at
        # the rightmost column

        # Track all positions in the traversal path
        positions = [0] * (tmp_x + tmp_y)
        positions[0] = (seed, -1)  # Store the starting
        # seed position
        position_tracker = 1

        # Traverse matrix until reaching the left edge (y=1)
        while tmp_y > 1:
            # Evaluate the three possible moves:
            # deletion, insertion, substitution/match
            # Use infinity for invalid moves (out of bounds)
            deletion_score = d[tmp_x - 1, tmp_y] if tmp_x >= 1 else np.inf
            insertion_score = d[tmp_x, tmp_y - 1] if tmp_y >= 1 else np.inf
            sub_or_match_score = (
                d[tmp_x - 1, tmp_y - 1] if tmp_x >= 1 and tmp_y >= 1 else np.inf
            )

            # Choose the best move - prioritize diagonal (match/substitution) when tied
            if (
                sub_or_match_score <= deletion_score
                and sub_or_match_score <= insertion_score
            ):
                # Check if we've already calculated the traversal for this position
                if trav_mat[tmp_x - 1][tmp_y - 1] != -1:
                    # If we have, we can short-circuit the calculation
                    start = trav_mat[tmp_x - 1][tmp_y - 1]
                    tmp_y = 1  # Force exit the while loop
                else:
                    # Diagonal move (substitution/match)
                    positions[position_tracker] = (tmp_x - 1, tmp_y - 1)
                    tmp_x -= 1
                    tmp_y -= 1
                    start = tmp_x
                    position_tracker += 1

            # Horizontal move (insertion into query) comes next in priority
            elif insertion_score <= deletion_score:
                if trav_mat[tmp_x][tmp_y - 1] != -1:
                    start = trav_mat[tmp_x][tmp_y - 1]
                    tmp_y = 1  # Force exit
                else:
                    positions[position_tracker] = (tmp_x, tmp_y - 1)
                    tmp_y -= 1
                    start = tmp_x
                    position_tracker += 1

            # Vertical move (deletion from query) as last option
            else:
                if trav_mat[tmp_x - 1][tmp_y] != -1:
                    start = trav_mat[tmp_x - 1][tmp_y]
                    tmp_y = 1  # Force exit
                else:
                    positions[position_tracker] = (tmp_x - 1, tmp_y)
                    tmp_x -= 1
                    start = tmp_x
                    position_tracker += 1

        # Record the start position for all cells in this path
        # This builds up a memoization table for future traversals
        for p in range(position_tracker):
            trav_mat[positions[p]] = start

        # Calculate vertical distance (how much of query is used)
        traversal_distance[seed] = abs(seed - trav_mat[(seed, -1)])

    return trav_mat, traversal_distance


def wagner_fisher_truncated(d):
    """
    Find the optimal sub-alignment from the Wagner-Fisher matrix.

    This function identifies the optimal end point in the last column that
    gives the best normalized edit distance, calculated as:
    (edit_distance + 1) / vertical_traversal_distance

    This approach favors alignments that match larger portions of the query
    sequence to the reference, which is particularly important for tRNA
    fragments where we want to identify the longest well-matching segment.

    Args:
        d: Dynamic programming matrix from wagner_fisher algorithm

    Returns:
        tuple: A tuple containing:
            - ndarray: Truncated matrix representing the optimal sub-alignment
            - int: Starting position in the reference sequence
            - int: Ending position in the reference sequence
    """
    # Traversal Distances
    trav, trav_dist = calcuate_vertical_traversal(d)

    scores = d[1:, -1] + 1
    scores = scores / trav_dist[1:]

    # Find the row with the minimum score
    truncate_idx = np.argmin(scores) + 1  # +1 because we skipped row 0

    # Return the distance at the selected point, the
    # truncated matrix, and the truncation point
    return d[: truncate_idx + 1, :], trav[(truncate_idx, -1)], truncate_idx
    # return d[truncate_idx -
    # trav[(truncate_idx,-1)]:truncate_idx+1,:],
    # trav[(truncate_idx,-1)], truncate_idx


def compute_edit_operations(s: str, t: str):
    """
    Compute the edit operations required to transform s (query) into t (reference).

    This function traces back through the alignment matrix to determine the
    specific sequence of operations (match, substitution, insertion, deletion)
    needed to transform the query into the reference. It leverages the truncated
    alignment to focus on the best matching region.

    Args:
        s: Query sequence (observed tRNA)
        t: Reference sequence (tRNA template)

    Returns:
        tuple: A tuple containing:
            - ndarray: Array of integer codes representing operations (ASCII values)
            - int: Starting position in the query
            - int: Ending position in the query

    Notes:
        The operation codes are ASCII values:
        - 'm' (109): Match
        - 's' (115): Substitution
        - 'i' (105): Insertion
        - 'd' (100): Deletion
    """
    # Get the distance matrix and truncate it
    _, full_matrix = wagner_fisher(s, t)
    d, start, stop = wagner_fisher_truncated(full_matrix)

    # We're Going to try a full truncation

    # Get dimensions of the truncated matrix
    m, n = d.shape[0] - 1, d.shape[1] - 1

    # Pre-allocate array with maximum possible size (m+n)
    max_operations = m + n
    instr_array = np.zeros(max_operations, dtype=np.int8)

    # Initialize index for filling the array (from the end)
    idx = max_operations - 1

    # Updated with AI generated comments:
    """
    # Trace back through the matrix
    while n > 0:
        deletion_score = d[m - 1, n] if m >= 1 else np.inf
        insertion_score = d[m, n - 1] if n >= 1 else np.inf
        sub_or_match_score = d[m - 1, n - 1] if m >= 1 and n >= 1 else np.inf

        if (sub_or_match_score <= deletion_score and
        sub_or_match_score <= insertion_score):
            if d[m - 1, n - 1] < d[m, n]:
                instr_array[idx] = ord('s')  # Substitution
            else:
                instr_array[idx] = ord('m')  # Match
            m -= 1
            n -= 1
        elif insertion_score <= deletion_score:
            # In terms of transforming query to reference:
            # Insertion into  query
            instr_array[idx] = ord('d')  # Deletion
            n -= 1
        else:
            # Deletion from query
            instr_array[idx] = ord('i')  # Insertion
            m -= 1

        idx -= 1
    #if d[m, n] == 1:
    #    instr_array[idx] = ord('s')
    #    idx -= 1
    #else:
    #    instr_array[idx] = ord('m')
    #    idx -= 1

    # Return only the relevant portion of the array
    return instr_array[idx+1:], start, stop
    """
    # Trace back through the matrix
    while n > 0:
        # For each position, evaluate all three possible previous moves
        # and choose the one that led to the current cell with minimum cost
        deletion_score = d[m - 1, n] if m >= 1 else np.inf
        insertion_score = d[m, n - 1] if n >= 1 else np.inf
        sub_or_match_score = d[m - 1, n - 1] if m >= 1 and n >= 1 else np.inf

        # Diagonal move: check if it's a match or substitution
        if (
            sub_or_match_score <= deletion_score
            and sub_or_match_score <= insertion_score
        ):
            if d[m - 1, n - 1] < d[m, n]:
                instr_array[idx] = ord("s")  # Substitution (mismatch)
            else:
                instr_array[idx] = ord("m")  # Match (identical bases)
            m -= 1
            n -= 1
        # Horizontal move: corresponds to deletion in reference frame
        elif insertion_score <= deletion_score:
            # Note: "deletion" here is relative to transforming query → reference
            # In alignment terms, this is an insertion into the query
            instr_array[idx] = ord("d")  # Deletion
            n -= 1
        # Vertical move: corresponds to insertion in reference frame
        else:
            # In alignment terms, this is a deletion from the query
            instr_array[idx] = ord("i")  # Insertion
            m -= 1

        idx -= 1

    return instr_array[idx + 1 :], start, stop


def edit_instructions(s: str, t: str):
    """
    Convert operation codes to human-readable edit instructions.

    This wrapper function transforms the numeric operation codes from
    compute_edit_operations into a more readable format as a list of
    character instructions.

    Args:
        s: Query sequence
        t: Reference sequence

    Returns:
        tuple: A tuple containing:
            - list: List of character edit operations ('m', 's', 'i', 'd')
            - int: Starting position in the query
            - int: Ending position in the query
    """
    operation_codes, start, stop = compute_edit_operations(s, t)

    # Convert codes to characters in a Python list
    instructions = [chr(code) for code in operation_codes]

    return instructions, start, stop


def cigar_tuples_from_edit_instrucitons(
    instructions, query_start, five_clip, query_end, three_clip
):
    """
    Convert edit instructions to CIGAR tuples compatible with SAM/BAM format.

    This function groups consecutive operations of the same type and
    handles soft clipping of unaligned portions at the 5' and 3' ends.
    The resulting CIGAR string represents the alignment in a format
    suitable for storage in BAM/SAM files.

    Args:
        instructions: List of edit operations ('m', 's', 'i', 'd')
        query_start: Position in query where alignment starts
        five_clip: Number of bases to soft-clip at the 5' end
        query_end: Position in query where alignment ends
        three_clip: Number of bases to soft-clip at the 3' end

    Returns:
        tuple: A tuple containing:
            - list: List of CIGAR tuples (operation_code, count)
            - int: Total edit distance for the alignment

    Notes:
        CIGAR operation codes:
        - 1: Insertion
        - 2: Deletion
        - 4: Soft clip
        - 7: Match
        - 8: Mismatch
    """
    cigar_tuples = []

    # Handle soft clipping at the start (query_start > 0)
    if max(query_start + five_clip, 0) > 0:
        cigar_tuples.append((4, query_start + five_clip))

    # Convert main alignment operations to CIGAR format
    # Group consecutive operations of the same type

    # AI comments:
    """
    prev_char = ""
    count = 0

    for op in instructions:
        # Map our internal operation codes to CIGAR operations
        if op == 'm':
            cigar_op = 7  # Match
        elif op == 's':
            cigar_op = 8  # Mismatch (also represented as 'X' in CIGAR)
        elif op == 'i':
            cigar_op = 1  # Insertion
        elif op == 'd':
            cigar_op = 2  # Deletion
        else:
            continue  # Skip unknown operations

        # If same operation as before, increment count
        if cigar_op == prev_char:
            count += 1
        else:
            # Add previous operation group to CIGAR string
            if prev_char != "":
                cigar_tuples.append((prev_char, count))
            # Start new operation group
            prev_char = cigar_op
            count = 1

    # Add the last operation group
    if prev_char:
        cigar_tuples.append((prev_char, count))

    if query_end + three_clip > 0:
        cigar_tuples.append((4, query_end + three_clip))

    edit_distance = 0
    for pair in cigar_tuples:
        if pair[0] != 4 and pair[0] != 7:
            edit_distance += pair[1]

    return cigar_tuples, int(edit_distance)
    """
    # Convert main alignment operations to CIGAR format
    # Group consecutive operations of the same type
    prev_char = ""
    count = 0

    for op in instructions:
        # Map our internal operation codes to CIGAR operations:
        # - 'm' → 7 (=): Sequence match
        # - 's' → 8 (X): Sequence mismatch
        # - 'i' → 1 (I): Insertion to the reference
        # - 'd' → 2 (D): Deletion from the reference
        if op == "m":
            cigar_op = 7  # Match (=)
        elif op == "s":
            cigar_op = 8  # Mismatch (X)
        elif op == "i":
            cigar_op = 1  # Insertion (I)
        elif op == "d":
            cigar_op = 2  # Deletion (D)
        else:
            continue  # Skip unknown operations

        # Group consecutive operations of the same type
        if cigar_op == prev_char:
            count += 1
        else:
            # Add previous operation group to CIGAR string
            if prev_char != "":
                cigar_tuples.append((prev_char, count))
            # Start new operation group
            prev_char = cigar_op
            count = 1

    # Add the last operation group
    if prev_char:
        cigar_tuples.append((prev_char, count))

    # Handle soft clipping at the end of the alignment
    if query_end + three_clip > 0:
        cigar_tuples.append((4, query_end + three_clip))

    # Calculate the total edit distance
    # Only count operations that aren't matches or soft clips
    edit_distance = 0
    for pair in cigar_tuples:
        if pair[0] != 4 and pair[0] != 7:  # If not soft clip (4) or match (7)
            edit_distance += pair[1]

    return cigar_tuples, int(edit_distance)


def identity_from_cigar(cigar_tuples):
    """
    Calculate the sequence identity percentage from a CIGAR string.

    Sequence identity is the proportion of matched positions to the
    total alignment length, excluding soft-clipped bases.

    Args:
        cigar_string: String representation of CIGAR operations

    Returns:
        float: Proportion of matched positions (0.0 to 1.0)
    """
    tuples = cigar_tuples
    total_align_length = 0
    matches = 0

    for count, op in tuples:
        if op == "S":  # Skip soft-clipped bases
            continue
        if op == "=":  # Count matches (Note: M includes both matches and mismatches)
            matches += count
            total_align_length += count
        else:  # Count other operations in alignment length
            total_align_length += count

    # Avoid division by zero
    if total_align_length == 0:
        return 0.0

    return matches / total_align_length


def subset_sequence(pysam_read, trna_indices):
    """
    Extract the tRNA portion of a sequence based on model inference indices.

    This function uses the move table (mv) and template start (ts) tags from
    a BAM/SAM record to identify the relevant portion of the read that
    corresponds to the predicted tRNA region from the inference model.

    Args:
        pysam_read: A pysam.AlignedSegment object (BAM/SAM record)
        trna_indices: Tuple of (start, end) positions predicted by the model

    Returns:
        tuple: A tuple containing:
            - str: Extracted tRNA sequence (empty if no valid region found)
            - int: Length of 3' unaligned region
            - int: Length of 5' unaligned region
    """
    # AI Comments:
    """
    mv_table = np.array(pysam_read.get_tag('mv'))
    ts = pysam_read.get_tag('ts')
    moves = np.where(mv_table == 1)[0]
    start = np.where((moves)*mv_table[0] + ts  >= trna_indices[0])
    stop = np.where((moves)*mv_table[0] + ts <= trna_indices[1])
    if len(start[0]) == 0 or len(stop[0]) == 0:
        return ("", None, None)
    start = start[0][0]
    stop = stop[0][-1]-1
    three_prime_slice = start
    five_prime_slice = len(moves) - stop

    reverse_sequence = pysam_read.query_sequence[::-1]
    return reverse_sequence[start:stop][::-1], three_prime_slice, five_prime_slice
    """
    # Extract the move table from the pysam read
    # The 'mv' tag contains a binary array indicating
    # where basecalls occurred (1=move, 0=stay)
    mv_table = np.array(pysam_read.get_tag("mv"))

    # Get the template start position from the 'ts' tag
    # This indicates where the alignment begins on the reference template
    ts = pysam_read.get_tag("ts")

    # Find all positions where a move was made (i.e., basecall positions)
    # This gives us the actual sequence positions corresponding to reference positions
    moves = np.where(mv_table == 1)[0]

    # Calculate the reference positions for each move and find which ones
    # fall within our target tRNA region:
    # - First multiply by mv_table[0] to handle strand orientation
    # - Then add template start to get absolute reference positions

    # Find the first position that's >= the tRNA start index
    start = np.where((moves) * mv_table[0] + ts >= trna_indices[0])

    # Find all positions that are <= the tRNA end index
    stop = np.where((moves) * mv_table[0] + ts <= trna_indices[1])

    # If we couldn't find valid start/stop positions, return empty results
    if len(start[0]) == 0 or len(stop[0]) == 0:
        return ("", None, None)

    # Get array indices for the first position after start and last position before end
    start = start[0][0]  # First base within tRNA region
    stop = stop[0][-1] - 1  # Last base within tRNA region

    # Calculate the length of unaligned regions for soft-clipping
    # These are important for generating correct CIGAR strings later
    three_prime_slice = start  # Number of bases before the tRNA region
    five_prime_slice = len(moves) - stop  # Number of bases after the tRNA region

    # The sequence is stored in reverse complement orientation in the read
    # So we first reverse it, extract the relevant portion, then reverse again
    # to get the correct orientation for the tRNA subsequence
    reverse_sequence = pysam_read.query_sequence[::-1]
    return reverse_sequence[start:stop][::-1], three_prime_slice, five_prime_slice


def align_read(pysam_read, inference_dict_read, ref_index, ref_sequence):
    """
    Create a new alignment for a read against a tRNA reference sequence.

    This function combines the subsetting, alignment, and CIGAR generation
    steps to produce a complete BAM/SAM record representing the alignment
    of the read to the reference tRNA sequence.

    Args:
        pysam_read: Original pysam.AlignedSegment object
        inference_dict_read: Dictionary with inference data
        including tRNA indices
        ref_index: Index of the reference sequence in the BAM header
        ref_sequence: The reference tRNA sequence to align against

    Returns:
        pysam.AlignedSegment: New BAM record with alignment information
    """
    # AI Comments:
    """
    a = pysam.AlignedSegment()
    if inference_dict_read['trna_indices'] == (-1, -1):
        return pysam_read
    sub_sequence, three_slice, five_slice =
    (subset_sequence(pysam_read,
    inference_dict_read['trna_indices'])
    if len(sub_sequence) == 0:
        return pysam_read

    a.query_name = pysam_read.query_name
    a.query_sequence = pysam_read.query_sequence
    a.query_qualities = pysam_read.query_qualities
    a.reference_id = ref_index
    a.flag = 0
    a.tags = pysam_read.get_tags()

    (edit_instruction_list,
     start,
     stop) = edit_instructions(sub_sequence,
                               ref_sequence)

    a.reference_start = 0
    a.mapping_quality = 60

    a.cigar, edit_dist =
cigar_tuples_from_edit_instrucitons(edit_instruction_list,
query_start=max(0, start-1), five_clip=five_slice,
query_end=len(sub_sequence) - stop,
three_clip=max(0,three_slice))

    a.set_tag('ED', edit_dist)
    return a
    """
    # Create a new empty alignment segment to store our results
    a = pysam.AlignedSegment()

    # Check if valid tRNA indices were identified by the inference model
    # If not (-1, -1), return the original read without any changes
    if inference_dict_read["trna_indices"] == (-1, -1):
        return pysam_read

    # Extract just the tRNA portion of the sequence using the predicted indices
    # Also get the lengths of unaligned regions (for soft clipping)
    sub_sequence, three_slice, five_slice = subset_sequence(
        pysam_read, inference_dict_read["trna_indices"]
    )

    # If no valid tRNA sequence could be extracted, return the original read
    if len(sub_sequence) == 0:
        return pysam_read

    # Transfer metadata from the original read to our new alignment
    a.query_name = pysam_read.query_name  # Read identifier
    a.query_sequence = pysam_read.query_sequence  # Full nucleotide sequence
    a.query_qualities = pysam_read.query_qualities  # Quality scores
    a.reference_id = ref_index  # Which reference sequence we're aligning to
    a.flag = 0  # No special SAM flags
    a.tags = pysam_read.get_tags()  # Preserve all original tags

    # Perform sequence alignment between the tRNA
    # subsequence and the reference
    # This calculates the edit operations needed and
    # identifies the best start/stop points
    (edit_instruction_list, start, stop) = edit_instructions(sub_sequence, ref_sequence)

    # Configure the alignment properties
    a.reference_start = 0  # Start at the beginning of the reference
    a.mapping_quality = 60  # High mapping quality (confident alignment)

    # Convert edit operations to CIGAR format for SAM/BAM compatibility
    # Also handle soft clipping of unaligned portions at both ends
    a.cigar, edit_dist = cigar_tuples_from_edit_instrucitons(
        edit_instruction_list,  # List of edit operations
        query_start=max(0, start - 1),  # Account for 0-based indexing in alignment
        five_clip=five_slice,  # Soft-clip bases at 5' end
        query_end=len(sub_sequence) - stop,  # Unaligned bases at end of alignment
        three_clip=max(0, three_slice),  # Soft-clip bases at 3' end
    )

    # Add a custom tag to store the edit distance (alignment quality metric)
    a.set_tag("ED", edit_dist)

    # Return the completed alignment record
    return a
