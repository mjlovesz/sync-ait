import hashlib
import json
import os

import numpy as np
from torch import nn

from msquickcmp.pta_acl_cmp.constant import AIT_DIALOG_DUMP_PATH


def dump_output_hook():
    infer_step = 0

    def hook_func(module, inputs, outputs):
        if not hasattr(module, "weight"):
            return outputs

        nonlocal infer_step
        w_md5 = hashlib.md5(module.weight.cpu().numpy().tobytes()).hexdigest()

        ait_dialog_dump_path = os.getenv(AIT_DIALOG_DUMP_PATH)
        ait_dialog_dump_path = "" or ait_dialog_dump_path
        pid = os.getpid()
        pid_dir = os.path.join(ait_dialog_dump_path, str(pid))
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


def register_hook(model, op_list=[]):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module.")
    if not isinstance(op_list, list):
        raise TypeError("op_list must be list.")
    for name, module in model.named_modules():
        if op_list:
            for op_type in op_list:
                if not isinstance(module, op_type):
                    continue
                module.name = name
                module.register_forward_hook(dump_output_hook())
        else:
            module.name = name
            module.register_forward_hook(dump_output_hook())


def set_dump_path(dump_path, dump_tag="ait_dump", backend="pt"):
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    dialog_path = os.path.join(dump_path, dump_tag)
    if not os.path.exists(dialog_path):
        os.mkdir(dialog_path)

    os.environ[AIT_DIALOG_DUMP_PATH] = dialog_path

    if backend == "acl":
        # TODO: set LD_PRELOAD
        pass
