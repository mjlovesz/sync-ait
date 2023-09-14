import hashlib
import json
import os

import numpy as np
from torch import nn


def dump_output_hook():
    infer_step = 0

    def hook_func(module, inputs, outputs):
        if not hasattr(module, "weight"):
            return outputs

        nonlocal infer_step
        w_md5 = hashlib.md5(module.weight.cpu().numpy().tobytes()).hexdigest()
        pid = os.getpid()
        cur_dir = os.getcwd()
        pid_dir = os.path.join(cur_dir, str(pid))
        if not os.path.exists(pid_dir):
            os.mkdir(pid_dir)

        token_dir = os.path.join(pid_dir, str(infer_step))
        if not os.path.exists(token_dir):
            os.mkdir(token_dir)

        out_data_path = os.path.join(token_dir, "{}_output.npy".format(module.name))
        np.save(out_data_path, outputs.cpu().numpy())

        metadata_path = os.path.join(pid_dir, "metadata.json")
        infer_step_key = str(infer_step)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as file:
                metadata = json.load(file)
            if metadata.get(infer_step_key):
                metadata.get(infer_step_key).setdefault(w_md5, [out_data_path])
            else:
                metadata.setdefault(infer_step_key, {w_md5: [out_data_path]})
        else:
            metadata = {infer_step_key: {w_md5: [out_data_path]}}

        with open(metadata_path, "w") as file:
            json.dump(metadata, file)

        infer_step += 1

    return hook_func


def register_hook(model):
    for name, module in model.named_modules():
        module.name = name
        module.register_forward_hook(dump_output_hook())
