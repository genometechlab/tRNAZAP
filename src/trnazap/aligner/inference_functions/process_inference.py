"""Module for processing tRNA model inference output."""
from splitter.storages.inference_results import InferenceResults

def load_inference_obj(inference_path_list):
    """
    Read in the pickled inference object(s).

    params:
        inference_path_list: list of paths to pickled results from the
        inference step
        ref_dict: Dictionary of numeric tRNA reference code keys with sequence
        and name information.

    return: A dictionary for each read in the dataset.
    """
    #For now just load one inference_object
    inference_obj = InferenceResults.load(inference_path_list)
    return inference_obj