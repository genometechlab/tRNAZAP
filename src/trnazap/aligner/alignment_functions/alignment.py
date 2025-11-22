"""Module that executes alignement functions.

This module has utility functions that can be applied
broadly to situtation that mimic the tRNA set up: Ionic
current based delineation of sequence encapsulating the
reference region.
"""

import numpy as np
import pysam
import numba
from numba import njit
numba.set_num_threads(1)

@njit(cache = True, fastmath = True)
def wagner_fisher_affine(s: str, t: str, gap_open = 2, gap_extend = 0.5):
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
    #gap_open = 2
    #gap_extend = 0.5
    
    m, n = len(s), len(t)
    d = np.zeros(shape=(m + 1, n + 1), dtype=np.float32)
    move_mat = np.zeros(shape=(m+1, n+1), dtype=np.int8)
    move_mat[0, 1:] = 2
    move_mat[1:, 0] = 1

    # Initialize first column with zeros to allow free start gaps in reference
    # This means the query can start aligning at any position in the reference
    d[0, :] = np.array([0] + [gap_open + gap_extend * (j-1) for j in range(1, n+1)])
    # Fill the matrix
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            
            if s[i - 1] == t[j - 1]:
                substitution_cost = d[i - 1, j - 1]
            else:
                substitution_cost = d[i - 1, j - 1] + 1
                
            if move_mat[i-1, j] == 1:
                deletion_score = d[i - 1, j] + gap_extend
            else:
                deletion_score = d[i - 1, j] + gap_open

            if move_mat[i, j-1] == 2:
                insertion_score = d[i, j-1] + gap_extend
            else:
                insertion_score = d[i, j-1] + gap_open

            if substitution_cost <= deletion_score and substitution_cost <= insertion_score:
                d[i, j] = substitution_cost
                if s[i-1] != t[j - 1]:
                    move_mat[i, j] = 3 # This is the signal value for a mismatch
            elif insertion_score <= deletion_score:
                move_mat[i, j] = 2 # Signal value for an insertion
                d[i, j] = insertion_score
            else:
                move_mat[i, j] = 1 # Signal value for a deletion 
                d[i, j] = deletion_score

    return move_mat, d


@njit(cache=True, fastmath=True)
def calculate_vertical_traversal(move_mat):
    """Optimized vertical traversal with reduced memory allocation"""
    rows, cols = move_mat.shape
    
    # Pre-allocate all arrays
    trav_mat = np.full((rows, cols), -1, dtype=np.int32)
    traversal_distance = np.full(rows, rows + cols + 1, dtype=np.int32)
    
    # Work backwards through seeds
    for seed in range(rows - 1, -1, -1):
        if trav_mat[seed, cols - 1] != -1:
            # Already computed
            continue
            
        # Track path using arrays instead of lists
        path_x = np.zeros(rows + cols, dtype=np.int32)
        path_y = np.zeros(rows + cols, dtype=np.int32)
        
        m, n = seed, cols - 1
        path_idx = 0
        path_x[path_idx] = m
        path_y[path_idx] = n
        path_idx += 1
        
        start = m

        while n > 0:
            # Diagonal move: check if it's a match or substitution
            if move_mat[m, n] == 0 or move_mat[m, n] == 3:
                m -= 1
                n -= 1
            # Horizontal move: corresponds to deletion in reference frame
            elif move_mat[m , n] == 2:
                # Note: "deletion" here is relative to transforming query → reference
                # In alignment terms, this is an insertion into the query
                n -= 1
            # Vertical move: corresponds to insertion in reference frame
            else:
                m -= 1

            if trav_mat[m, n] != -1:
                start = trav_mat[m, n]
                break
            
            start = m
            path_x[path_idx] = m
            path_y[path_idx] = n
            path_idx += 1

        # Update trav_mat for all positions in path
        for i in range(path_idx):
            trav_mat[path_x[i], path_y[i]] = start
        
        traversal_distance[seed] = abs(seed - start)
    
    return trav_mat, traversal_distance

def wagner_fisher_truncated(dist_mat, move_mat):
    """
    Find the optimal sub-alignment from the Wagner-Fisher matrix.

    This function identifies the optimal end point in the last column that
    gives the best normalized edit distance, calculated as:
    (edit_distance + 1) / vertical_traversal_distance

    This approach favors alignments that match larger portions of the query
    sequence to the reference, which is particularly important for tRNA
    fragments where we want to identify the longest well-matching segment.

    Args:
        d: Dynamic programming transition matrix from wagner_fisher algorithm

    Returns:
        tuple: A tuple containing:
            - ndarray: Truncated matrix representing the optimal sub-alignment
            - int: Starting position in the reference sequence
            - int: Ending position in the reference sequence
    """
    # Traversal Distances
    trav, trav_dist = calculate_vertical_traversal(move_mat)

    scores = dist_mat[1:, -1] + 1
    with np.errstate(divide='ignore'):
        scores = scores / trav_dist[1:]

    # Find the row with the minimum score
    truncate_idx = np.argmin(scores) + 1  # +1 because we skipped row 0

    # Return the distance at the selected point, the
    # truncated matrix, and the truncation point
    return move_mat[: truncate_idx + 1, :], trav[(truncate_idx, -1)], truncate_idx

def compute_edit_operations_affine(s: str, t: str, gap_open = 2, gap_extend = 0.5):
    move_mat, dist_mat = wagner_fisher_affine(s, t, gap_open, gap_extend)
    move_mat, start, stop = wagner_fisher_truncated(dist_mat, move_mat)
    
    # Get dimensions of the truncated matrix
    m, n = move_mat.shape[0] - 1, move_mat.shape[1] - 1

    # Pre-allocate array with maximum possible size (m+n)
    max_operations = m + n
    instr_array = np.zeros(max_operations, dtype=np.int8)

    # Initialize index for filling the array (from the end)
    idx = max_operations - 1

    # Trace back through the matrix
    while n > 0:
        # Diagonal move: check if it's a match or substitution
        if move_mat[m, n] == 0:
            instr_array[idx] = ord("m")  # Match (identical bases)
            m -= 1
            n -= 1
        elif move_mat[m, n] == 3:
            instr_array[idx] = ord("s") 
            m -= 1
            n -= 1
        # Horizontal move: corresponds to deletion in reference frame
        elif move_mat[m , n] == 2:
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

def edit_instructions(s: str, t: str, wf_gap_open = 2.0, wf_gap_extend = 0.5,):
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
    operation_codes, start, stop = compute_edit_operations_affine(s, t, wf_gap_open, wf_gap_extend)

    # Convert codes to characters in a Python list
    instructions = [chr(code) for code in operation_codes]

    return instructions, start, stop


def cigar_tuples_from_edit_instrucitons(
    instructions, query_start, five_clip, query_end, three_clip, numeric_code=False
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
        cigar_tuples.append((4, int(query_start + five_clip)))

    # Convert main alignment operations to CIGAR format
    # Group consecutive operations of the same type
    prev_char = ""
    count = 0

    for op in instructions:
        
        if not numeric_code:
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
        else:
            cigar_op = op

        # Group consecutive operations of the same type
        if cigar_op == prev_char:
            count += 1
        else:
            # Add previous operation group to CIGAR string
            if prev_char != "":
                cigar_tuples.append((int(prev_char), int(count)))
            # Start new operation group
            prev_char = cigar_op
            count = 1

    # Add the last operation group
    if prev_char:
        cigar_tuples.append((int(prev_char), int(count)))

    # Handle soft clipping at the end of the alignment
    if query_end + three_clip > 0:
        cigar_tuples.append((4, int(query_end + three_clip)))

    # Calculate the total edit distance
    # Only count operations that aren't matches or soft clips
    edit_distance = 0
    for pair in cigar_tuples:
        if pair[0] != 4 and pair[0] != 7:  # If not soft clip (4) or match (7)
            edit_distance += pair[1]
    #print(f"{query_start=} {five_clip=} {query_end=} {three_clip=} {cigar_tuples=} {instructions=}")
    return cigar_tuples, int(edit_distance)

#Could be njit if we preprocess the pysam read parts?
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
    start = max(start[0][0]-1, 0)  # First base within tRNA region
    #stop = stop[0][-1] - 1  # Last base within tRNA region
    #testing why the -1 was there
    stop = stop[0][-1]+1

    # Calculate the length of unaligned regions for soft-clipping
    # These are important for generating correct CIGAR strings later
    three_prime_slice = start  # Number of bases before the tRNA region
    five_prime_slice = max(0, len(moves) - stop)  # Number of bases after the tRNA region

    # The sequence is stored in reverse complement orientation in the read
    # So we first reverse it, extract the relevant portion, then reverse again
    # to get the correct orientation for the tRNA subsequence
    reverse_sequence = pysam_read.query_sequence[::-1]
    return reverse_sequence[start:stop][::-1], three_prime_slice, five_prime_slice

def check_fragment(cigar, reference_length, min_del_proportion=0.15):
    """
    Docstring
    """
    
    del_count = sum([x[1] for x in cigar if x[0] == 2])
    if del_count/reference_length >= min_del_proportion:
        return True
    return False

@njit(cache=True, fastmath=True)
def smith_waterman_for_fragment(tRNA, fragment, gap_open = -6, gap_extend = -1, match_score = +3, mismatch_score = -1):
    """Perform Smith-Waterman local alignment with affine gap penalties.
    
    Simplified implementation that uses the traceback matrix itself to track
    gap states, implementing affine penalties without needing separate state matrices.
    
    Args:
        tRNA (str): The reference sequence (typically the longer sequence)
        fragment (str): The query sequence to align against the tRNA
    
    Returns:
        tuple: (score_mat, traceback_mat, length_mat, tRNA_start, frag_start)
    """
    # Affine gap scoring parameters
    #gap_open = -6       # Penalty for opening a new gap (higher)
    #gap_extend = -1     # Penalty for extending an existing gap (lower)
    #match_score = +3    # Reward for matching bases
    #mismatch_score = -1 # Penalty for mismatched bases
    min_matches = 25 #Only consider scores for regions with 25 matches
    
    # Get sequence lengths
    tRNA_len = len(tRNA)
    fragment_len = len(fragment)
    
    # Initialize matrices
    score_mat = np.zeros((tRNA_len+1, fragment_len+1), dtype=np.float32)
    traceback_mat = np.full((tRNA_len+1, fragment_len+1), -1, dtype=np.int8)
    length_mat = np.zeros((tRNA_len+1, fragment_len+1), dtype=np.int32)
    

    
    # Track best overall score and position
    max_score = 0
    tRNA_start = 0
    frag_start = 0
    
    # Fill matrices using dynamic programming
    for i in range(1, tRNA_len + 1):
        for j in range(1, fragment_len + 1):
            # Option 1: Match/Mismatch (diagonal move)
            is_match = tRNA[i-1] == fragment[j-1]
            if is_match:
                diagonal_score = score_mat[i-1][j-1] + match_score
            else:
                diagonal_score = score_mat[i-1][j-1] + mismatch_score
            
            # Option 2: Deletion (horizontal move - gap in tRNA)
            # Check if the cell we're coming from was reached by a deletion
            if traceback_mat[i][j-1] == 1:  # Already in a deletion
                deletion_score = score_mat[i][j-1] + gap_extend
            else:  # Opening a new deletion
                deletion_score = score_mat[i][j-1] + gap_open
            
            # Option 3: Insertion (vertical move - gap in fragment)
            # Check if the cell we're coming from was reached by an insertion
            if traceback_mat[i-1][j] == 2:  # Already in an insertion
                insertion_score = score_mat[i-1][j] + gap_extend
            else:  # Opening a new insertion
                insertion_score = score_mat[i-1][j] + gap_open
            
            # Choose the best option
            best_score = max(0, diagonal_score, deletion_score, insertion_score)
            score_mat[i][j] = best_score
            
            # Update traceback based on best option
            if best_score == 0:
                traceback_mat[i][j] = -1
                length_mat[i][j] = 0
            elif best_score == diagonal_score:
                # Match/mismatch
                if is_match:
                    traceback_mat[i][j] = 7  # Match
                    length_mat[i][j] = length_mat[i-1][j-1] + 1
                else:
                    traceback_mat[i][j] = 8  # Mismatch
                    length_mat[i][j] = length_mat[i-1][j-1]
            elif best_score == deletion_score:
                # Deletion (gap in tRNA)
                traceback_mat[i][j] = 1
                length_mat[i][j] = length_mat[i][j-1]
            else:
                # Insertion (gap in fragment)
                traceback_mat[i][j] = 2
                length_mat[i][j] = length_mat[i-1][j]
            
            # Track overall best score
            if best_score >= max_score:
                max_score = best_score
                tRNA_start = i
                frag_start = j
    
    return score_mat, traceback_mat, length_mat, tRNA_start, frag_start
    
def edit_instructions_from_smith_waterman(traceback_mat, max_len, tRNA_start, frag_start):
    """Extract edit instructions from a Smith-Waterman traceback matrix.
    
    Traces back through the alignment path from the highest-scoring position to the
    start of the local alignment, collecting the edit operations needed to transform
    the reference sequence (tRNA) into the query sequence (fragment). Uses a 
    reverse-filling strategy to avoid needing to reverse the instruction array.
    
    The function uses extended CIGAR operation codes:
    - 7: Match (= in extended CIGAR)
    - 8: Mismatch (X in extended CIGAR)
    - 2: Deletion (D in CIGAR) - base in reference but not in query
    - 1: Insertion (I in CIGAR) - base in query but not in reference
    - -1: Stop signal indicating start of alignment
    
    Args:
        traceback_mat (np.ndarray): The traceback matrix from Smith-Waterman algorithm
                                   containing operation codes at each position.
        max_len (int): Maximum expected length of the edit instruction array.
                      Should be at least tRNA_start + frag_start for safety.
        tRNA_start (int): Row index in the matrix where alignment ends (and traceback begins).
                         Note: This is the END of the alignment, not the start.
        frag_start (int): Column index in the matrix where alignment ends (and traceback begins).
                         Note: This is the END of the alignment, not the start.
    
    Returns:
        tuple: A 3-element tuple containing:
            - edit_instructions (list): Array of operation codes in forward order,
                                      representing the sequence of edits from alignment
                                      start to end.
            - frag_end (int): Column index where alignment begins (0-indexed sequence position).
                             Despite the variable name, this is actually the START of alignment.
            - tRNA_end (int): Row index where alignment begins (0-indexed sequence position).
                             Despite the variable name, this is actually the START of alignment.
    
    Raises:
        ValueError: If an unknown traceback code is encountered or if the traceback
                   path exceeds the allocated buffer size.
    
    Note:
        The naming convention here is potentially confusing: the input positions are
        called 'start' but represent the END of the alignment (where we start traceback),
        while the returned positions are called 'end' but represent the START of the
        alignment (where traceback ends). This is a historical artifact of the
        traceback perspective versus the alignment perspective.
    """
    # Pre-allocate array for edit instructions
    # Using empty strings as placeholders - will be replaced with operation codes
    edit_instructions = [""] * max_len
    
    # Start filling from the end of the array using negative indexing
    # This technique means our final array will already be in forward order
    instruction_pos = -1
    
    # Initialize traceback position at the highest-scoring cell
    # These represent our current position in the matrix during traceback
    i, j = tRNA_start, frag_start
    
    # Trace back through the matrix until we hit a stop signal or matrix boundary
    # The three conditions protect against different scenarios:
    # - traceback_mat[i,j] != -1: Stop when we hit the alignment start marker
    # - i > 0: Prevent going past the first row (empty tRNA prefix)
    # - j > 0: Prevent going past the first column (empty fragment prefix)
    while traceback_mat[i,j] != -1 and i > 0 and j > 0:
        # Store the operation code at the current position
        # Using negative indexing fills the array from right to left
        edit_instructions[instruction_pos] = traceback_mat[i,j]
        instruction_pos -= 1  # Move to the previous position in our buffer
        
        # Move to the previous cell based on the operation type
        # The movement direction tells us which sequences consumed bases
        
        if traceback_mat[i,j] == 1:  # Insertion operation
            # Fragment has an extra base relative to tRNA
            # Move left in matrix (only fragment position decreases)
            j -= 1
            
        elif traceback_mat[i,j] == 2:  # Deletion operation
            # tRNA has an extra base relative to fragment  
            # Move up in matrix (only tRNA position decreases)
            i -= 1
            
        elif traceback_mat[i,j] == 7 or traceback_mat[i,j] == 8:  # Match or Mismatch
            # Both sequences have bases at this position
            # Move diagonally in matrix (both positions decrease)
            i -= 1
            j -= 1
            
        else:
            # Encountered an unexpected operation code
            # This indicates either data corruption or a bug in the Smith-Waterman implementation
            raise ValueError(
                f"Unknown traceback code {traceback_mat[i,j]} encountered at "
                f"matrix position ({i}, {j}). Valid codes are: -1 (stop), "
                f"3 (deletion), 4 (insertion), 7 (match), 8 (mismatch)."
            )
    
    # Safety check: ensure our traceback path didn't exceed the allocated buffer
    # If this triggers, max_len was too small for the alignment
    if instruction_pos < -max_len:
        raise ValueError(
            f"Traceback path length ({-instruction_pos} operations) exceeded "
            f"allocated buffer size ({max_len}). Consider increasing max_len "
            f"to at least {-instruction_pos}."
        )
    
    # Return the used portion of the edit instructions array
    # instruction_pos+1 because instruction_pos points to the last unused position
    # Also return the final positions, which represent the alignment START
    # (confusingly named 'end' when put in dynamic programming context)
    return edit_instructions[instruction_pos+1:], i, j

def fragment_align(sub_sequence, 
                   ref_sequence, 
                   five_clip, 
                   three_clip,    
                   sw_gap_open = -6.0,
                   sw_gap_extend = -1.0,
                   sw_match = 3.0,
                   sw_mismatch = 1.0,):
    """Align a fragment sequence to a reference sequence and generate CIGAR string.
    
    This function orchestrates the complete alignment pipeline: running Smith-Waterman,
    extracting the alignment path, and converting it to standard CIGAR format. The
    coordinate naming can be confusing due to the traceback perspective.
    
    Args:
        sub_sequence (str): The query sequence (fragment) to be aligned
        ref_sequence (str): The reference sequence (e.g., tRNA) to align against
        five_clip (int): Number of bases clipped from 5' end of fragment (for CIGAR soft-clipping)
        three_clip (int): Number of bases clipped from 3' end of fragment (for CIGAR soft-clipping)
    
    Returns:
        tuple: (cigar, edit_dist, tRNA_start, tRNA_end, frag_start, frag_end)
            - cigar: CIGAR string or tuples describing the alignment
            - edit_dist: Edit distance (number of mismatches + indels)
            - tRNA_start: Where alignment ENDS in reference (matrix coordinates)
            - tRNA_end: Where alignment STARTS in reference (matrix coordinates)
            - frag_start: Where alignment ENDS in fragment (matrix coordinates)
            - frag_end: Where alignment STARTS in fragment (matrix coordinates)
        Returns (None, None, None, None, None, None) if alignment quality is too low
    
    Note on coordinate naming:
        Due to the traceback perspective, 'start' variables indicate where we START
        the traceback (which is the END of the alignment), while 'end' variables
        indicate where we END the traceback (which is the START of the alignment).
        This is counterintuitive but maintained for consistency with the underlying functions.
    """

    # Step 1: Perform Smith-Waterman local alignment
    # This builds three matrices: scores, traceback operations, and match counts
    # IMPORTANT: tRNA_start and frag_start represent the HIGHEST SCORING position,
    # which is where the alignment ENDS (not where it starts!)
    score_mat, traceback_mat, length_mat, tRNA_start, frag_start = smith_waterman_for_fragment(
        ref_sequence,   # Reference sequence (rows in matrix)
        sub_sequence,    # Query sequence (columns in matrix)
        sw_gap_open,
        sw_gap_extend,
        sw_match,
        sw_mismatch,
    )
    
    # Step 2: Quality control - filter out poor alignments
    # The length_mat tracks number of matches, requiring at least 25 matches
    # helps avoid reporting spurious short alignments that occur by chance

    # Check if any position met the minimum match requirement
    if tRNA_start == 0 and frag_start == 0:
        # No position had >= min_matches
        return None, None, None, None, None, None
    
    if length_mat[tRNA_start, frag_start] < 25:
        # Return None for all values to indicate alignment failure
        return None, None, None, None, None, None
    
    # Step 3: Extract the alignment path by tracing back through the matrix
    # This walks backwards from the highest scoring position to find where
    # the alignment began, collecting edit operations along the way
    # CONFUSING: The returned tRNA_end and frag_end are where traceback ENDS,
    # which means where the alignment actually STARTS!
    edit_instructions, tRNA_end, frag_end = edit_instructions_from_smith_waterman(
        traceback_mat,                            # Matrix of operation codes
        len(ref_sequence) + len(sub_sequence),    # Maximum possible alignment length
        tRNA_start,                               # Row where we begin traceback (alignment end)
        frag_start                                # Column where we begin traceback (alignment end)
    )

    query_end = len(sub_sequence) - frag_start
    # Step 4: Convert edit instructions to standard CIGAR format
    # CIGAR strings are the standard way to represent alignments in bioinformatics
    # The five_clip and three_clip parameters are used to add soft-clipping operations
    # to the CIGAR string for bases that were trimmed before alignment
    # Note: There's a typo in the function name (instrucitons -> instructions)
    cigar, edit_dist = cigar_tuples_from_edit_instrucitons(
        edit_instructions,  # Array of operation codes from traceback
        frag_end,           # End position in fragment (for boundary handling)
        five_clip,          # Bases clipped from 5' end (will add S operation to CIGAR)
        query_end,          # Start position in fragment (for boundary handling)
        three_clip,         # Bases clipped from 3' end (will add S operation to CIGAR)
        numeric_code=True   # Return numeric CIGAR codes instead of letters
    )

    cigar, ref_shift, edit_offset = trim_cigar_to_first_match_window(
        cigar, 
        window_size=8, 
        min_matches=6
    )
    
    # Adjust edit distance and reference position based on trimming
    edit_dist = edit_dist - edit_offset
    tRNA_end_adjusted = tRNA_end + ref_shift
    
    return cigar, edit_dist, tRNA_start, tRNA_end_adjusted, frag_start, frag_end

@njit(cache=True, fastmath=True)
def find_first_match_in_window(expanded, start_offset, window_size, min_matches):
    """Find the index of the first '=' in the first window with >= min_matches.
    
    Returns -1 if no good window found.
    """
    total_length = len(expanded)
    
    for i in range(start_offset, total_length - window_size + 1):
        # Count matches in window
        match_count = 0
        
        for j in range(window_size):
            if expanded[i + j] == 7:  # Match
                match_count += 1
        
        if match_count >= min_matches:
            # Found good window, now find first '=' within it
            for j in range(window_size):
                if expanded[i + j] == 7:
                    return i + j
    
    return -1

def trim_cigar_to_first_match_window(cigar_tuples, window_size=8, min_matches=6):
    """Fast trimming that finds first window with min_matches '=' ops, then trims to first '=' in that window.
    
    Args:
        cigar_tuples: List of (operation, length) tuples (pysam format)
        window_size: Size of sliding window (default 8)
        min_matches: Minimum number of '=' ops required in window (default 6)
    
    Returns:
        tuple: (trimmed_cigar, ref_start_shift, total_changes_to_soft)
    """
    if not cigar_tuples:
        return cigar_tuples, 0, 0
    
    # Check for leading soft clip and skip past it
    start_offset = 0
    if cigar_tuples[0][0] == 4:  # Leading soft clip
        start_offset = cigar_tuples[0][1]
    
    # Expand CIGAR once (as numpy array for numba)
    total_length = sum(length for _, length in cigar_tuples)
    expanded = np.empty(total_length, dtype=np.uint8)
    pos = 0
    for op, length in cigar_tuples:
        expanded[pos:pos+length] = op
        pos += length
    
    if total_length - start_offset < window_size:
        return [], 0, 0
    
    # Use numba to find first match index
    first_match_idx = find_first_match_in_window(expanded, start_offset, window_size, min_matches)
    
    if first_match_idx == -1:
        return [], 0, 0
    
    # Count what we're trimming (everything before first_match_idx)
    delta_ref_start = 0
    soft_clip_bases = 0
    delta_edit_dist = 0
    
    for i in range(first_match_idx):
        op = expanded[i]
        if op == 2:  # Deletion - consumes reference
            delta_ref_start += 1
            delta_edit_dist +=1
        elif op == 8:  # Mismatch - consumes both
            delta_ref_start += 1
            soft_clip_bases += 1
            delta_edit_dist += 1
        elif op == 1: #Insertion
            delta_edit_dist += 1
            soft_clip_bases += 1
        elif op == 4: #Soft Clip
            soft_clip_bases += 1
        elif op == 7: #Match
            soft_clip_bases += 1
            delta_ref_start += 1
    
    
    # Build new CIGAR starting from first_match_idx
    result = []
    
    # Add soft clip for everything we trimmed
    if soft_clip_bases > 0:
        result.append((4, soft_clip_bases))
    
    # Compress remaining operations back into CIGAR tuples
    current_op = int(expanded[first_match_idx])
    current_count = 1
    
    for i in range(first_match_idx + 1, total_length):
        if expanded[i] == current_op:
            current_count += 1
        else:
            result.append((current_op, current_count))
            current_op = int(expanded[i])
            current_count = 1
    
    # Don't forget the last operation
    result.append((current_op, current_count))
    
    return result, delta_ref_start, delta_edit_dist

def trim_cigar_to_matches(cigar_tuples):
    """Trim CIGAR to start and end with matches.
    
    Any trimmed operations that consume query bases (mismatches, insertions,
    soft clips) are converted to soft clips to maintain the correct query length.
    
    Example:
        Input:  [(4,5), (2,6), (7,1), (2,1), (7,3), (2,32), (4,45)]
                  5S      6D      1=      1D      3=      32D      45S
        Output: [(4,5), (7,1), (2,1), (7,3), (4,45)], ref_shift=6, counts=(0,0,38)
        
    The 6D at start is removed and ref_start shifts by 6.
    The 32D at end is removed (no ref_start change).
    
    Args:
        cigar_tuples: List of (operation, length) tuples (pysam format)
        
    CIGAR operation codes:
        0 = M (alignment match - could be match or mismatch)
        1 = I (insertion)
        2 = D (deletion)
        4 = S (soft clipping)
        7 = = (sequence match, extended CIGAR)
        8 = X (sequence mismatch, extended CIGAR)
    
    Returns:
        tuple: (trimmed_cigar, ref_start_shift, mismatches_to_soft, insertions_to_soft, deletions_removed)
            - trimmed_cigar: Updated CIGAR list with soft clips
            - ref_start_shift: How many bases to add to reference_start
            - total_changes_to_soft: Number of operations converted to soft clips
    """
    if not cigar_tuples:
        return cigar_tuples, 0, 0
    
    # Initialize counts
    mismatches_to_soft = 0
    insertions_to_soft = 0
    deletions_removed = 0
    
    # Find first match
    first_match = -1
    for i, (op, length) in enumerate(cigar_tuples):
        if op in (0, 7):  # Match (M or =)
            first_match = i
            break
    
    if first_match == -1:  # No matches found
        return [], 0, 0
    
    # Find last match
    last_match = -1
    for i in range(len(cigar_tuples) - 1, -1, -1):
        if cigar_tuples[i][0] in (0, 7):  # Match (M or =)
            last_match = i
            break
    
    # Check if trimming is even needed
    needs_trimming = False
    
    # Check start: anything before first match that's not a soft clip?
    for i in range(first_match):
        if cigar_tuples[i][0] != 4:  # Not a soft clip
            needs_trimming = True
            break
    
    # Check end: anything after last match that's not a soft clip?
    for i in range(last_match + 1, len(cigar_tuples)):
        if cigar_tuples[i][0] != 4:  # Not a soft clip
            needs_trimming = True
            break
    
    if not needs_trimming:
        return cigar_tuples, 0, 0
    
    # Count reference bases consumed before first match (deletions only)
    # Also count ALL query bases that need to be soft-clipped
    ref_start_shift = 0
    extra_soft_clip_5 = 0
    for i in range(first_match):
        op, length = cigar_tuples[i]
        if op == 2:  # Deletion
            ref_start_shift += length
            deletions_removed += length
        elif op == 1:  # Insertion
            extra_soft_clip_5 += length
            insertions_to_soft += length
        elif op == 4:  # Soft clip
            extra_soft_clip_5 += length
        elif op == 8:  # Mismatch
            ref_start_shift += length
            extra_soft_clip_5 += length
            mismatches_to_soft += length
    
    # Count ALL query bases after last match that need to be soft-clipped
    extra_soft_clip_3 = 0
    for i in range(last_match + 1, len(cigar_tuples)):
        op, length = cigar_tuples[i]
        if op == 2:  # Deletion
            deletions_removed += length
        elif op == 1:  # Insertion
            extra_soft_clip_3 += length
            insertions_to_soft += length
        elif op == 4:  # Soft clip
            extra_soft_clip_3 += length
        elif op == 8:  # Mismatch
            extra_soft_clip_3 += length
            mismatches_to_soft += length
    
    # Build final CIGAR
    result = []
    if extra_soft_clip_5 > 0:
        result.append((4, extra_soft_clip_5))  # Soft clip
    result.extend(cigar_tuples[first_match:last_match + 1])
    if extra_soft_clip_3 > 0:
        result.append((4, extra_soft_clip_3))  # Soft clip
    
    return result, ref_start_shift, mismatches_to_soft + insertions_to_soft + deletions_removed

@njit(fastmath = True, cache = True)
def compare_shot_in_the_dark(result2_ed, result2_start, result2_end, result1_ed, result1_start, result1_end):
    if result2_start - result2_end <= 0:
        return False
    elif result1_start - result1_end <= 0:
        return True
    if ((result2_start - result2_end - result2_ed) / (result2_start - result2_end) > 
        (result1_start - result1_end - result1_ed) / (result1_start - result1_end)):
        return True
    return False

@njit(fastmath = True, cache = True)
def frag_update(pre_cigar_first, pre_cigar_last, post_cigar_first, post_cigar_last):
    three_shift = 0
    five_shift = 0
    
    if pre_cigar_first[0] == 4 and post_cigar_first[0] == 4:
        three_shift += post_cigar_first[1] - pre_cigar_first[1]

    if pre_cigar_first[0] != 4 and post_cigar_first[0] == 4:
        three_shift += post_cigar_first[1]

    if pre_cigar_last[0] == 4 and post_cigar_last[0] == 4:
        five_shift += post_cigar_last[1] - pre_cigar_last[1]

    if pre_cigar_last[0] != 4 and post_cigar_last[0] == 4:
        five_shift += post_cigar_last[1]

    return (five_shift, three_shift)

@njit(fastmath = True, cache = True)
def ident_from_cigar(cigar_tuples):
    matches = 0
    mismatches = 0
    insertions = 0
    deletions = 0
    for tup in cigar_tuples:
        if tup[0] == 7:
            matches += tup[1]
        elif tup[0] == 8:
            mismatches += tup[1]
        elif tup[0] == 1:
            insertions += tup[1]
        elif tup[0] == 2:
            deletions += tup[1]
    return matches / (matches+mismatches+insertions+deletions)

#This sucks. Would it be faster to make all three reads and just check them? Prolly more accurate...
def shot_in_the_dark_alignment(pysam_read, 
                               top_three_ref_dict, 
                               ident_threshold,
                               sw_gap_open = -6.0,
                               sw_gap_extend = -1.0,
                               sw_match = 3.0,
                               sw_mismatch = 1.0,):
    #Top three ref dict:
    # {index : [ref_index, ref_sequence]}
    # This will be 0, 1, 2 based on the order of classification
    pre_results = [fragment_align(pysam_read.query_sequence, 
                                  top_three_ref_dict[i][1], 
                                  0, 
                                  0, 
                                  sw_gap_open, 
                                  sw_gap_extend, 
                                  sw_match, 
                                  sw_mismatch) for i in range(3)]
    results = []
    for pr in pre_results:
        if pr[0] is None or len(pr[0])==0:  # Skip failed alignments
            results.append((None, float('inf'), 0, 0, 0, 0))
            continue

        results.append(pr)                      
    #Each element of results contains the following:
        # (cigar, edit_dist, tRNA_start, tRNA_end, frag_start, frag_end)
    
    best_index = 0
    best_result = results[0]
    if (abs(best_result[3] - best_result[2]) < 25 or 
        compare_shot_in_the_dark(results[1][1], 
                                 results[1][3], 
                                 results[1][2], 
                                 best_result[1], 
                                 best_result[3], 
                                 best_result[2])):
        best_result = results[1]
        best_index = 1
    if (abs(best_result[3] - best_result[2]) < 25 or 
        compare_shot_in_the_dark(results[2][1], 
                                 results[2][3], 
                                 results[2][2], 
                                 best_result[1], 
                                 best_result[3], 
                                 best_result[2])):
        best_result = results[2]
        best_index = 2
    
    if best_result[0] is None:
        return pysam_read
        
    a = pysam.AlignedSegment()
    a.query_name = pysam_read.query_name               # Unique read identifier
    a.query_sequence = pysam_read.query_sequence       # Complete original sequence (not just tRNA part)
    a.query_qualities = pysam_read.query_qualities     # Phred quality scores for each base
    a.reference_id = top_three_ref_dict[best_index][0] # Which reference this aligns to
    a.flag = 0                                         # Start with unmapped flag (will update if secondary)
    a.tags = pysam_read.get_tags()                     # Preserve any custom tags (RG, BC, etc.)
    a.mapping_quality = 3 - best_index
    a.set_tag('ls', best_index)
    a.reference_start = best_result[3]
    a.cigar = best_result[0]
    a.set_tag("ED", best_result[1])
    
    if (ident_from_cigar(best_result[0])) < ident_threshold or a.get_cigar_stats()[0][7] < 25:
        return pysam_read
    else:
        return a    

def align_read(
    pysam_read, 
    inference_dict_read, 
    ref_index, 
    ref_sequence, 
    wf_gap_open = 2.0, 
    wf_gap_extend = 0.5, 
    sw_gap_open = -6.0,
    sw_gap_extend = -1.0,
    sw_match = 3.0,
    sw_mismatch = 1.0,
    ident_threshold = 0.75,
    secondary=False
):
    """Create a new alignment for a sequencing read against a tRNA reference sequence.
    
    This function represents the core of a tRNA alignment pipeline. It takes a sequencing
    read that potentially contains tRNA sequences (as predicted by a machine learning model)
    and creates a proper alignment against a known tRNA reference. The process involves
    extracting the relevant portion of the read, performing sequence alignment, and
    packaging the results in standard BAM format.
    
    Args:
        pysam_read (pysam.AlignedSegment): Original read from a BAM/SAM file. This may be
            unaligned or aligned to a different reference. Contains the full sequence data,
            quality scores, and metadata.
        inference_dict_read: ML model output containing predicted tRNA boundaries within
            the read. Key attribute is 'variable_region_range' - a tuple of (start, end)
            positions indicating where the tRNA sequence is located.
        ref_index (int): Numerical index of the reference sequence in the BAM header.
            This tells downstream tools which reference sequence this read aligns to.
        ref_sequence (str): The actual nucleotide sequence of the reference tRNA.
            This is what we'll align the extracted portion of the read against.
        secondary (bool): If True, marks this as a secondary alignment (SAM flag 256).
            Used when a read might align to multiple tRNA references.
    
    Returns:
        pysam.AlignedSegment: New alignment record with updated CIGAR string and
            coordinates. Returns original read unchanged if alignment fails.
    """
    
    # Create a fresh AlignedSegment object to build our new alignment
    # We don't modify the original because we might need to preserve it or
    # create multiple alignments from the same read
    a = pysam.AlignedSegment()
    
    # Check if the ML model successfully identified a tRNA region in this read
    # The value (-1, -1) is a sentinel indicating "no tRNA found"
    #print(inference_dict_read[1])
    
    if inference_dict_read[1] == (-1, -1): #SHOT IN THE DARK ALIGNMENT
        # No predicted tRNA region - return the original read unchanged
        # This preserves any existing alignment information
        #print("skipped")
        return pysam_read


    ts = pysam_read.get_tag('ts')
    if ts > inference_dict_read[1][0]:
        tRNA_signal_indices = (ts, inference_dict_read[1][1])
    else:
        tRNA_signal_indices = inference_dict_read[1]
    # Extract the predicted tRNA portion from the full read sequence
    # This function returns:
    # - sub_sequence: just the tRNA bases
    # - three_slice: number of bases after the tRNA (3' end)
    # - five_slice: number of bases before the tRNA (5' end)
    # These flanking regions will be soft-clipped in the final alignment
    sub_sequence, three_slice, five_slice = subset_sequence(
        pysam_read, tRNA_signal_indices
    )
    
    #print(f"{pysam_read.query_sequence=}")
    #print(f"{sub_sequence=}\t{three_slice=}\t{five_slice=}")
    # Sanity check: ensure we actually extracted something
    # This could fail if the predicted boundaries were invalid
    if len(sub_sequence) == 0: #SHOT IN THE DARK ALIGNMENT
        return pysam_read
    
    # Transfer all the read metadata to our new alignment record
    # This preserves important information for downstream analysis
    a.query_name = pysam_read.query_name          # Unique read identifier
    a.query_sequence = pysam_read.query_sequence  # Complete original sequence (not just tRNA part)
    a.query_qualities = pysam_read.query_qualities # Phred quality scores for each base
    a.reference_id = ref_index                    # Which reference this aligns to
    a.flag = 0                                    # Start with unmapped flag (will update if secondary)
    a.tags = pysam_read.get_tags()                # Preserve any custom tags (RG, BC, etc.)
    a.set_tag('ZP',inference_dict_read[1])
    a.set_tag('ZI',(int(five_slice), int(three_slice)))
    # Perform the actual sequence alignment between extracted tRNA and reference
    # This function (not shown) likely uses dynamic programming to find the best
    # alignment and returns:
    # - edit_instruction_list: sequence of operations (match/mismatch/insertion/deletion)
    # - start: where alignment begins in the query
    # - stop: where alignment ends in the query
    (edit_instruction_list, start, stop) = edit_instructions(sub_sequence, ref_sequence, wf_gap_open, wf_gap_extend)
    
    # Set the alignment position on the reference
    # Starting at 0 means we're aligning to the beginning of the reference tRNA
    # This assumes the reference is a complete tRNA sequence, not a larger chromosome
    a.reference_start = 0
    
    # Set mapping quality to 60 (on Phred scale)
    # This indicates high confidence in the alignment (probability of error ~10^-6)
    # Standard values: 60 = uniquely mapped, 0 = unmapped, 1-59 = varying confidence
    a.mapping_quality = 60
    
    # Handle secondary alignment flagging
    # Flag 256 in SAM specification indicates "secondary alignment"
    # This is used when a read could align equally well to multiple references
    if secondary:
        a.mapping_quality = 0  
        a.flag = 256           # Set the secondary alignment bit flag
    
    # Convert our alignment operations into standard CIGAR format
    # CIGAR strings encode alignments compactly: M=match/mismatch, I=insertion, D=deletion, S=soft clip
    # The soft clipping (S operations) indicates bases present in the read but not part of alignment
    cigar, edit_dist = cigar_tuples_from_edit_instrucitons(
        edit_instruction_list,
        query_start=max(0, start),
        five_clip=five_slice,
        query_end=len(sub_sequence) - stop,
        three_clip=max(0, three_slice),
    )

    # Use the new fast window-based trimming for ALL alignments
    trimmed_cigar, ref_off_set, edit_delta = trim_cigar_to_first_match_window(
        cigar,
        window_size=8,
        min_matches=6
    )
    
    edit_dist = edit_dist - edit_delta
    a.cigar = trimmed_cigar
    a.reference_start += ref_off_set
    a.set_tag("ED", edit_dist)

    if a.reference_length is None:
        return pysam_read
        
    # Fragment alignment fallback stays the same
    if check_fragment(a.cigar, len(ref_sequence)):
        # fragment_align already does its own trimming internally
        cigar, edit_dist, ref_start, ref_stop, query_start, query_stop = fragment_align(
            sub_sequence, 
            ref_sequence, 
            five_slice, 
            three_slice, 
            sw_gap_open, 
            sw_gap_extend,
            sw_match,
            sw_mismatch,
        )
        
        if (cigar is None or edit_dist > (ref_start - ref_stop) * 0.3):
            return pysam_read
        
        a.set_tag("FG", 1)
        a.set_tag("ED", edit_dist)
        a.reference_start = ref_stop
        a.cigar = cigar
    
    elif ident_from_cigar(a.cigartuples) < ident_threshold or a.get_cigar_stats()[0][7] < 25:
        return pysam_read
        
    return a
