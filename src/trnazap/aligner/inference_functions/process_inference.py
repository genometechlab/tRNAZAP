"""Module for processing tRNA model inference output."""
from splitter.storages.inference_results import InferenceResults
from splitter import ModelConfig, ModelLoader, Inference, ResultsVisualizer, ZIRReader
import torch
import tqdm

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

    inference_obj={}
    for pth in inference_path_list:
        with ZIRReader(pth, index=False) as zip_reader:
            lbls_to_cls = zip_reader.metadata.label_names
            for read_result in tqdm.tqdm(zip_reader):
                variable_region = read_result.variable_region_range
                cls_ = lbls_to_cls[str(read_result.classification_pred)]
                inference_obj[read_result.read_id] = (cls_, variable_region)
            if len(inference_obj) >= 1000000:
                break
    return inference_obj