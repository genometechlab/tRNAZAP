import glob
import torch
import os
import os.path as osp
import torch.nn as nn


def load_weights(model, path):
    state_dict = torch.load(path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("Missing keys:")
        for key in missing_keys:
            print(f"{key}: {state_dict[key].shape if key in state_dict else 'N/A'}")
    if unexpected_keys:
        print("Unexpected keys:")
        for key in unexpected_keys:
            print(key)
    return model
