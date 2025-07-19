"""Module that executes alignement functions.

This module has utility functions that can be applied
broadly to situtation that mimic the tRNA set up: Ionic
current based delineation of sequence encapsulating the
reference region.
"""

import numpy as np
import pysam
from numba import njit


@njit(parallel=True)
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
    with np.errstate(divide='ignore'):
        scores = scores / trav_dist[1:]

    # Find the row with the minimum score
    truncate_idx = np.argmin(scores) + 1  # +1 because we skipped row 0

    # Return the distance at the selected point, the
    # truncated matrix, and the truncation point
    return d[: truncate_idx + 1, :], trav[(truncate_idx, -1)], truncate_idx


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
    #stop = stop[0][-1] - 1  # Last base within tRNA region
    #testing why the -1 was there
    stop = stop[0][-1]

    # Calculate the length of unaligned regions for soft-clipping
    # These are important for generating correct CIGAR strings later
    three_prime_slice = start  # Number of bases before the tRNA region
    five_prime_slice = len(moves) - stop  # Number of bases after the tRNA region

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

@njit(parallel=True)
def smith_waterman_for_fragment(tRNA, fragment):
    """Perform Smith-Waterman local alignment between a tRNA and fragment sequence.
    
    Implements the Smith-Waterman dynamic programming algorithm to find the optimal
    local alignment between two sequences. This implementation uses extended CIGAR
    format codes that distinguish between matches and mismatches, providing more
    detailed alignment information than standard CIGAR strings.
    
    The algorithm builds three matrices:
    - score_mat: Tracks alignment scores at each position
    - traceback_mat: Records which operation led to each score (for path reconstruction)
    - length_mat: Counts the number of matching positions in the alignment
    
    Args:
        tRNA (str): The reference sequence (typically the longer sequence).
                   In alignment terms, this is treated as the "reference".
        fragment (str): The query sequence to align against the tRNA.
                       Gaps in this sequence are insertions, gaps in tRNA are deletions.
    
    Returns:
        tuple: A 5-element tuple containing:
            - score_mat (np.ndarray): Complete scoring matrix of shape (tRNA_len+1, fragment_len+1)
            - traceback_mat (np.ndarray): Matrix of traceback operations using extended CIGAR codes:
                * -1: Stop/Start of alignment
                * 7: Match (equivalent to '=' in extended CIGAR)
                * 8: Mismatch (equivalent to 'X' in extended CIGAR)
                * 3: Deletion (equivalent to 'D' in CIGAR) - base present in tRNA but not fragment
                * 4: Insertion (equivalent to 'I' in CIGAR) - base present in fragment but not tRNA
            - length_mat (np.ndarray): Matrix tracking number of matches in optimal path to each cell
            - tRNA_start (int): Matrix row index where optimal alignment ends (1-indexed)
            - frag_start (int): Matrix column index where optimal alignment ends (1-indexed)
    
    Note:
        The returned positions (tRNA_start, frag_start) represent the END of the optimal
        alignment in matrix coordinates. To get the actual sequence positions, subtract 1
        from these values. The START of the alignment must be found through traceback.
    """
    # Scoring parameters for the alignment algorithm
    # These values determine how the algorithm weighs matches, mismatches, and gaps
    gap = -2            # Penalty for introducing a gap (insertion or deletion)
    match_score = +3    # Reward for matching bases
    mismatch_score = -1 # Penalty for mismatched bases
    
    # Get sequence lengths for matrix dimensioning
    tRNA_len = len(tRNA)
    fragment_len = len(fragment)
    
    # Initialize the scoring matrix with zeros
    # Extra row and column (hence +1) represent alignment with empty sequence
    score_mat = np.zeros((tRNA_len+1, fragment_len+1), dtype = np.int32)
    
    # Initialize traceback matrix with -1 (stop signal)
    # This matrix records which operation gave the optimal score for each cell
    # Using extended CIGAR format codes for more detailed alignment info:
    # -1 = stop/start of alignment
    # 7 = match ('=' in extended CIGAR)
    # 8 = mismatch ('X' in extended CIGAR)  
    # 3 = deletion ('D' in CIGAR) - base in tRNA but not in fragment
    # 4 = insertion ('I' in CIGAR) - base in fragment but not in tRNA
    traceback_mat = np.full((tRNA_len+1, fragment_len+1), -1, dtype = np.int8)
    
    # Initialize matrix to track number of matching positions
    # This helps evaluate alignment quality beyond just the score
    length_mat = np.zeros((tRNA_len+1, fragment_len+1), dtype = np.int32)
    
    # Variables to track the cell with maximum score (best local alignment)
    max_score = 0
    tRNA_start = 0  # Will store row index of best alignment end
    frag_start = 0  # Will store column index of best alignment end
    
    # Fill the matrices using dynamic programming
    # i represents position in tRNA, j represents position in fragment
    for i in range(1, tRNA_len + 1):
        for j in range(1, fragment_len + 1):
            # Check if current positions match
            # Note: i-1 and j-1 because matrix is 1-indexed but sequences are 0-indexed
            is_match = False
            if tRNA[i - 1] == fragment[j - 1]:
                # Calculate score if we align these matching bases
                diagonal_score = score_mat[i-1][j-1] + match_score  
                is_match = True
            else:
                # Calculate score if we align these mismatching bases
                diagonal_score = score_mat[i-1][j-1] + mismatch_score
            
            # Calculate score if we introduce a gap in tRNA (deletion from fragment's perspective)
            # This means we're coming from the left cell (j-1)
            deletion_score = score_mat[i][j-1] + gap
            
            # Calculate score if we introduce a gap in fragment (insertion from fragment's perspective)
            # This means we're coming from the cell above (i-1)
            insertion_score = score_mat[i-1][j] + gap
            
            # Choose the best scoring option
            # The order of these conditions implements a tie-breaking strategy:
            # diagonal > deletion > insertion when scores are equal
            if (diagonal_score >= deletion_score and 
                diagonal_score >= insertion_score and
                diagonal_score >= 0):  # Smith-Waterman: can start new alignment if all options are negative
                
                score_mat[i][j] = diagonal_score
                
                if is_match:
                    traceback_mat[i][j] = 7  # Extended CIGAR '=' for match
                    length_mat[i][j] = length_mat[i-1][j-1] + 1  # Increment match count
                else:
                    traceback_mat[i][j] = 8  # Extended CIGAR 'X' for mismatch
                    length_mat[i][j] = length_mat[i-1][j-1]  # Preserve match count (no increment)
            
            elif (deletion_score >= insertion_score and
                  deletion_score >= 0):
                
                score_mat[i][j] = deletion_score
                traceback_mat[i][j] = 2  # CIGAR 'D' - deletion relative to reference
                # Coming from left cell, so copy its match count
                length_mat[i][j] = length_mat[i][j-1]
            
            elif (insertion_score >= 0):
                
                score_mat[i][j] = insertion_score
                traceback_mat[i][j] = 1  # CIGAR 'I' - insertion relative to reference
                # Coming from above cell, so copy its match count
                length_mat[i][j] = length_mat[i-1][j]
            
            else:
                # All options are negative - start a new alignment here
                score_mat[i][j] = 0
                traceback_mat[i][j] = -1  # Mark as potential start position
                length_mat[i][j] = 0      # New alignment has no matches yet
            
            # Track the highest scoring cell (best local alignment found so far)
            # In case of ties, this keeps the last (bottom-right-most) occurrence
            if score_mat[i][j] >= max_score:
                max_score = score_mat[i][j]
                tRNA_start = i  # Remember this is matrix coordinates (1-indexed)
                frag_start = j  # Will need to subtract 1 for sequence position
                
    return score_mat, traceback_mat, length_mat, tRNA_start, frag_start
    
def edit_instructions_from_smith_waterman(traceback_mat, max_len, tRNA_start, frag_start):
    """Extract edit instructions from a Smith-Waterman traceback matrix.
    
    Traces back through the alignment path from the highest-scoring position to the
    start of the local alignment, collecting the edit operations needed to transform
    the reference sequence (tRNA) into the query sequence (fragment). Uses a clever
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
    # This clever technique means our final array will already be in forward order
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
    # (confusingly named 'end' for historical reasons)
    return edit_instructions[instruction_pos+1:], i, j

def fragment_align(sub_sequence, ref_sequence, five_clip, three_clip):
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
        sub_sequence    # Query sequence (columns in matrix)
    )
    
    # Step 2: Quality control - filter out poor alignments
    # The length_mat tracks number of matches, requiring at least 10 matches
    # helps avoid reporting spurious short alignments that occur by chance
    if length_mat[tRNA_start, frag_start] < 10:
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
    
    # Step 5: Return all alignment information
    # Remember the coordinate naming confusion:
    # - tRNA_start, frag_start: Where alignment ENDS (highest score position)
    # - tRNA_end, frag_end: Where alignment STARTS (traceback end position)
    # This is backwards from intuition but consistent with the implementation
    return cigar, edit_dist, tRNA_start, tRNA_end, frag_start, frag_end



def align_read(
    pysam_read, inference_dict_read, ref_index, ref_sequence, secondary=False
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
    if inference_dict_read[1] == (-1, -1):
        # No predicted tRNA region - return the original read unchanged
        # This preserves any existing alignment information
        #print("skipped")
        return pysam_read
    
    # Extract the predicted tRNA portion from the full read sequence
    # This function returns:
    # - sub_sequence: just the tRNA bases
    # - three_slice: number of bases after the tRNA (3' end)
    # - five_slice: number of bases before the tRNA (5' end)
    # These flanking regions will be soft-clipped in the final alignment
    sub_sequence, three_slice, five_slice = subset_sequence(
        pysam_read, inference_dict_read[1]
    )
    #print(f"{pysam_read.query_sequence=}")
    #print(f"{sub_sequence=}\t{three_slice=}\t{five_slice=}")
    # Sanity check: ensure we actually extracted something
    # This could fail if the predicted boundaries were invalid
    if len(sub_sequence) == 0:
        return pysam_read
    
    # Transfer all the read metadata to our new alignment record
    # This preserves important information for downstream analysis
    a.query_name = pysam_read.query_name          # Unique read identifier
    a.query_sequence = pysam_read.query_sequence  # Complete original sequence (not just tRNA part)
    a.query_qualities = pysam_read.query_qualities # Phred quality scores for each base
    a.reference_id = ref_index                    # Which reference this aligns to
    a.flag = 0                                    # Start with unmapped flag (will update if secondary)
    a.tags = pysam_read.get_tags()                # Preserve any custom tags (RG, BC, etc.)
    
    # Perform the actual sequence alignment between extracted tRNA and reference
    # This function (not shown) likely uses dynamic programming to find the best
    # alignment and returns:
    # - edit_instruction_list: sequence of operations (match/mismatch/insertion/deletion)
    # - start: where alignment begins in the query
    # - stop: where alignment ends in the query
    (edit_instruction_list, start, stop) = edit_instructions(sub_sequence, ref_sequence)
    
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
        a.mapping_quality = 60  # Keep high quality even for secondary alignments
        a.flag = 256           # Set the secondary alignment bit flag
    
    # Convert our alignment operations into standard CIGAR format
    # CIGAR strings encode alignments compactly: M=match/mismatch, I=insertion, D=deletion, S=soft clip
    # The soft clipping (S operations) indicates bases present in the read but not part of alignment
    a.cigar, edit_dist = cigar_tuples_from_edit_instrucitons(
        edit_instruction_list,              # The alignment operations to encode
        query_start=max(0, start - 1),      # Convert to 0-based coordinates (alignment uses 1-based)
        five_clip=five_slice,               # Bases before tRNA to soft-clip (5' end)
        query_end=len(sub_sequence) - stop, # Calculate bases after alignment to soft-clip
        three_clip=max(0, three_slice),     # Bases after tRNA to soft-clip (3' end)
    )
    
    # Store the edit distance as a custom SAM tag
    # Edit distance = number of mismatches + number of indels in the alignment
    # This is a standard metric for alignment quality (lower is better)
    a.set_tag("ED", edit_dist)
    
    # Fragment alignment check - this is a fallback strategy
    # Some tRNA sequences might only partially overlap with the reference
    # (e.g., if the read contains a tRNA fragment rather than complete tRNA)
    if check_fragment(a.cigar, len(ref_sequence)):
        # Try a more permissive fragment-based alignment approach
        # This might find a better local alignment for partial sequences
        cigar, edit_dist, ref_start, ref_stop, query_start, query_stop = fragment_align(
            sub_sequence, ref_sequence, five_slice, three_slice
        )
        
        if cigar is None:
            return pysam_read
            
        
        # If fragment alignment succeeded and is better than our original alignment
        # (lower edit distance = better alignment), use it instead
        elif edit_dist / ((ref_start-ref_stop)) < a.get_tag("ED")/(abs(a.reference_start - a.reference_end)):
            # Mark this as a fragment alignment with a custom tag
            a.set_tag("FG", 1)
            a.set_tag("ED", edit_dist)
            # IMPORTANT: ref_stop here is actually where the alignment STARTS
            # This is due to the traceback perspective in the fragment_align function
            # where 'stop' means where we stop tracing back (= start of alignment)
            a.reference_start = ref_stop
            # Replace with the better CIGAR from fragment alignment
            a.cigar = cigar
    
    
    # Return the completed alignment record, ready for output to BAM/SAM
    return a
