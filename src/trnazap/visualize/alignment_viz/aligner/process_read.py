import numpy as np
from numba import njit

@njit
def positional_array(read_ref_end, 
                     read_ref_start, 
                     read_align_end, 
                     aligned_pairs, 
                     ref_seq, 
                     read_seq,
                     region_start, 
                     region_end):
    """
    Create tracking array with 4 rows: matches, insertions, coverage, deletions.
    
    Returns:
    --------
    track_arr : np.array of shape (4, length) where:
        [0] = match counts at each position
        [1] = insertion counts at each position
        [2] = coverage (1 if position covered, 0 otherwise)
        [3] = deletion counts at each position
    """
    # Use pysam's reference_start and reference_end for efficient overlap check
    if read_ref_end <= region_start or read_ref_start >= region_end:
        return np.empty((4, 0), dtype=np.float64)

    # 4 rows now: matches, insertions, coverage, deletions
    track_arr = np.full((4, (region_end - region_start)), np.nan)
    track_arr[1] = 0  # Insertion count
    track_arr[2] = 0  # Coverage
    track_arr[3] = 0  # Deletion count
    
    # Track the most recent reference position we've seen
    last_ref_pos = -1
    
    # Get aligned pairs (query_pos, ref_pos)
    for query_pos, ref_pos in aligned_pairs:   
        if query_pos != -1 and query_pos >= read_align_end:
            break
        if ref_pos == -1:  # Insertion
            if last_ref_pos == -1:
                continue
            idx = last_ref_pos - region_start
            # Only count insertion if it's within our region
            if last_ref_pos != -1 and region_start <= last_ref_pos < region_end:
                track_arr[1][idx] += 1
            continue
        last_ref_pos = ref_pos
        if ref_pos < region_start or ref_pos >= region_end:
            continue

        idx = ref_pos - region_start
        if np.isnan(track_arr[0][idx]):
            track_arr[0][idx] = 0
            
        if query_pos == -1:  # Deletion - bases in reference skipped by read
            track_arr[3][idx] += 1  
            continue
        else:
            # Check if it's a match or mismatch
            query_base = read_seq[query_pos]
            ref_base = ref_seq[ref_pos]
            
            if query_base == ref_base:
                track_arr[0][idx] += 1
    
    track_arr[2] = (~np.isnan(track_arr[0])).astype(np.float64)
    return track_arr

@njit
def read_pass(track_arr, include_insertions=True, ident_threshold=0.75, min_coverage=25):
    """
    Determine if read passes quality thresholds.
    Works with 4-row track_arr.
    
    Returns:
    --------
    ident : float
        Identity score
    passes : bool
        Whether read passes thresholds
    alignment_length : float
        Number of reference positions covered
    """
    total = np.count_nonzero(~np.isnan(track_arr[0]))
    if include_insertions:
        total += np.nansum(track_arr[1])
    matches = np.nansum(track_arr[0])
    if total == 0:
        return 0.0, False, 0
        
    ident = matches / total
    alignment_length = np.nansum(track_arr[2])

    return ident, (ident >= ident_threshold) & (np.nansum(track_arr[2]) >= min_coverage), alignment_length