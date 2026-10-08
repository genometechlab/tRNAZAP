import glob
import torch
import os
import os.path as osp
import torch.nn as nn


def load_weights(model, path):
    """Load a checkpoint into ``model``, failing on any architecture mismatch.

    Frozen parameters (e.g. the identity stem's fixed conv kernel) are rebuilt at
    init, so checkpoints are allowed to omit them. Any other missing or unexpected
    key raises, so a head/config mismatch can't leave layers randomly initialized.
    """
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device, weights_only=True)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if k not in frozen]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {path} does not match the model built from the config.\n"
            f"  Missing keys: {missing}\n"
            f"  Unexpected keys: {result.unexpected_keys}\n"
            "Check head_type, tasks and the architecture fields in the config."
        )
