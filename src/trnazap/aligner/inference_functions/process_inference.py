"""Module for processing tRNA model inference output."""
from splitter.storages.inference_results import InferenceResults
from splitter import ModelConfig, ModelLoader, Inference, ResultsVisualizer, ZIRReader
import torch
import os
import numpy as np
from tqdm import tqdm


#TODO: Fix the logic for loading files

def load_inference_obj(inference_path):
    """
    Read in the pickled inference object(s).

    params:
        inference_path: Either a dir with .zir files or a single .zir file
        
    return: A dictionary for each read in the dataset.
    """

    if os.path.isdir(inference_path[0]):
        return from_dir(inference_path[0])
    else:
        return from_files(inference_path)

def from_files(inference_path_list):
    inference_obj={}
    for pth in inference_path_list:
        print(f"Loading Single File: {pth}")
        with ZIRReader(pth, index=False) as zip_reader:
            lbls_to_cls = zip_reader.metadata.label_names
            for read_result in zip_reader:
                variable_region = read_result.variable_region_range
                cls_ = lbls_to_cls[str(read_result.classification_pred)]
                sorted_classes = np.argsort(read_result.classification_probs)
                secondary_cls = lbls_to_cls[str(sorted_classes[-2])]
                tertiary_cls = lbls_to_cls[str(sorted_classes[-3])]
                fragment = str(read_result.fragmentation_pred)
                inference_obj[read_result.read_id] = (cls_, variable_region, secondary_cls, tertiary_cls, fragment)
    return inference_obj

def from_dir(dir_path):
    files = [f for f in os.listdir(dir_path) if f[-4:] == ".zir"]
    print(f"Loading inference files in: {dir_path}")
    inference_obj = {}
    for i, pth in tqdm(enumerate(files)):
        pth = os.path.join(dir_path, pth)
        print(pth)
        with ZIRReader(pth, index=False) as zip_reader:
            lbls_to_cls = zip_reader.metadata.label_names
            for read_result in zip_reader:
                variable_region = read_result.variable_region_range
                cls_ = lbls_to_cls[str(read_result.classification_pred)]
                sorted_classes = np.argsort(read_result.classification_probs)
                secondary_cls = lbls_to_cls[str(sorted_classes[-2])]
                tertiary_cls = lbls_to_cls[str(sorted_classes[-3])]
                fragment = str(read_result.fragmentation_pred)
                inference_obj[read_result.read_id] = (cls_, variable_region, secondary_cls, tertiary_cls, fragment)
    return inference_obj
        
        