# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json
import os

import numpy as np
from torch import nn

import msquickcmp
from msquickcmp.pta_acl_cmp.constant import AIT_DUMP_PATH, AIT_IS_SAVE_MD5, AIT_DIALOG_DUMP_PATH

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def dump_output_hook():
    infer_step = 0

    def hook_func(module, inputs, outputs):
        if not hasattr(module, "weight"):
            pass

        nonlocal infer_step
        w_md5 = hashlib.md5(module.weight.cpu().numpy().tobytes()).hexdigest()

        ait_dump_path = os.getenv(AIT_DUMP_PATH)
        ait_dump_path = ait_dump_path or ""
        pid = os.getpid()
        pid_dir = os.path.join(ait_dump_path, str(pid))
        if not os.path.exists(pid_dir):
            os.mkdir(pid_dir)

        token_dir = os.path.join(pid_dir, str(infer_step))
        if not os.path.exists(token_dir):
            os.mkdir(token_dir)

        out_data_path = os.path.abspath(os.path.join(token_dir, "{}_output.npy".format(module.name)))
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

        with os.fbopen(os.open(metadata_path, WRITE_FLAGS, WRITE_MODES), "w") as file:
            json.dump(metadata, file)

        infer_step += 1
        return outputs

    return hook_func


def register_hook(model, op_list):
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


def set_dump_path(dump_path=".", dump_tag="ait_dump", backend="pt"):
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    dialog_path = os.path.join(dump_path, dump_tag)
    if not os.path.exists(dialog_path):
        os.mkdir(dialog_path)

    os.environ[AIT_DIALOG_DUMP_PATH] = dialog_path

    if isinstance(backend, str) and backend.lower() == "acl":
        os.environ[AIT_IS_SAVE_MD5] = "1"       
