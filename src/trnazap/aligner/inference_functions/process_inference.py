"""Module for processing tRNA model inference output."""
import torch
import os
import numpy as np
from tqdm import tqdm
import pickle
import sys

#TODO: Fix the logic for loading files

def load_inference_obj(inference_path, threads=1, thread_index=0, pickled=False):
    """
    Read in the pickled inference object(s).
    params:
        inference_path: Either a dir with .zir files or a single .zir file or list of files
        
    return: A dictionary for each read in the dataset.
    """
    from ...io import ZIRReader
    from ...storages import ReadResult, ReadResultCompressed

    if pickled:
        with open(inference_path[0], 'rb') as infile:
            return pickle.load(infile)
    
    # Handle single path vs list
    if isinstance(inference_path, (list, tuple)):
        if len(inference_path) == 1:
            inference_path = inference_path[0]
    
    # ZIRReader handles both single files and directories automatically
    with ZIRReader(inference_path, index=False) as reader:
        lbls_to_cls = reader.metadata.label_names
        inference_obj = {}
        
        for read_result in tqdm(reader, 
                                total=len(reader),
                                disable = (threads - 1 != thread_index),
                                leave=False,
                                desc="Processing reads"):
            if int(read_result.read_id[:8], 16) % threads != thread_index:
                continue
            if isinstance(read_result, ReadResultCompressed):
                # Compressed record - use top3_classes directly
                top3_sorted = read_result.top3_classes  # np.ndarray with top 3 class indices
                
                # Convert to strings for label lookup
                pred_str = str(top3_sorted[0])
                secondary_str = str(top3_sorted[1])
                tertiary_str = str(top3_sorted[2])
                
                # Lookup labels
                cls_ = lbls_to_cls[pred_str]
                secondary_cls = lbls_to_cls[secondary_str]
                tertiary_cls = lbls_to_cls[tertiary_str]
                
                # Use fragmented field
                fragment_str = str(read_result.fragmented)
                
            else:  # isinstance(read_result, ReadResult)
                # Full ReadResult - compute from classification_probs
                probs = read_result.classification_probs
                
                # Get indices of 3 largest values (unsorted within themselves)
                top3_indices = np.argpartition(probs, -3)[-3:]
                
                # Sort only these 3 indices by their probability values
                top3_sorted = top3_indices[np.argsort(probs[top3_indices])][::-1]
                
                # Convert to strings once
                pred_str = str(top3_sorted[0])
                secondary_str = str(top3_sorted[1])
                tertiary_str = str(top3_sorted[2])
                
                # Lookup labels
                cls_ = lbls_to_cls[pred_str]
                secondary_cls = lbls_to_cls[secondary_str]
                tertiary_cls = lbls_to_cls[tertiary_str]
                
                # Use fragmentation_pred
                fragment_str = str(read_result.fragmentation_pred)
            
            inference_obj[read_result.read_id] = (
                cls_,
                read_result.variable_region_range,
                secondary_cls,
                tertiary_cls,
                fragment_str
            )
    
    return inference_obj

if __name__ == "__main__":
    inf_obj = load_inference_obj(sys.argv[1])

    with open(sys.argv[2], 'wb') as outfile:
        pickle.dump(inf_obj, outfile)
    
        
        