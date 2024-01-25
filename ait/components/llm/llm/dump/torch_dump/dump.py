# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import os

import torch
import numpy as np


def singleton(cls):
    ins = {}

    def run(*args, **kwargs):
        if cls not in ins:
            ins[cls] = cls(*args, **kwargs)
        return ins.get(cls)

    return run


@singleton
class DumpConfig:
    def __init__(self, dump_path=None, mode=None, token_range=None,
                 module_list=None, api_list=None, tensor_part=None, device_id=None):
        self.dump_path = dump_path or "./"
        self.mode = mode or "module"
        self.token_range = token_range or [0]
        self.module_list = module_list or []
        self.api_list = api_list or []
        self.tensor_part = tensor_part
        self.dump_flag = True
        self.token_id = 0
        self.module_ids = {}
        self.cur_module_id = 0
        self.device_id = device_id or 0
        self.dump_dir = os.path.join(self.dump_path, "{}_{}".format(str(os.getpid()), str(self.device_id)))

    def update_module_ids(self, module_name):
        self.cur_module_id += 1
        if module_name not in self.module_ids:
            self.module_ids[module_name] = self.cur_module_id


def dump_tensor(feat, feat_path):
    if isinstance(feat, (tuple, list)):
        for idx, tensor in enumerate(feat):
            dump_tensor(tensor, "{}_{}".format(feat_path, idx))
    elif isinstance(feat, torch.Tensor):
        data = feat.cpu().detach().numpy()
        if not feat_path.endswith(".npy"):
            feat_path = feat_path + ".npy"
        np.save(feat_path, data)


def dump_data(inputs, outputs, dump_path, exec_count, tensor_part):
    if tensor_part == "0":
        dump_tensor(inputs, os.path.join(dump_path, "output_exec" + str(exec_count)))
    elif tensor_part == "1":
        dump_tensor(outputs, os.path.join(dump_path, "input_exec" + str(exec_count)))
    else:
        dump_tensor(inputs, os.path.join(dump_path, "input_exec" + str(exec_count)))
        dump_tensor(outputs, os.path.join(dump_path, "output_exec" + str(exec_count)))


def dump_module_hook():
    exec_count = 0

    def hook_func(module: torch.nn.Module, inputs, outputs):
        nonlocal exec_count
        dump_config = DumpConfig()
        if dump_config.mode == "api":  # 仅dump api的数据
            return
        # 只dump指定module的数据
        if dump_config.module_list and not isinstance(module, dump_config.module_list):
            return

        if dump_config.token_id == 0:
            dump_config.update_module_ids(module.name)

        if not dump_config.dump_flag:
            return

        exec_count += 1
        module_name = module.name
        dump_path = os.path.join(dump_config.dump_path, dump_config.token_id, module_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

        dump_data(inputs, outputs, dump_path, exec_count, dump_config.tensor_part)
        # 将dump_config.module_ids传给方锴的update接口，将模型树状信息保存成json文件。
        if module.name == "root" and dump_config.token_id == 0:
            import json
            with open("module_ids.json", "w") as file:
                json.dump(dump_config.module_ids, file)

    return hook_func


def set_dump_flag():
    cur_token_id = 0

    def hook_func(module, inputs):
        nonlocal cur_token_id
        config = DumpConfig()
        config.token_id = cur_token_id
        # 通过root module执行的轮次来判断当前在第几个token
        if config.token_id in config.token_range:
            config.dump_flag = True
        else:
            config.dump_flag = False

        cur_token_id += 1
        return

    return hook_func
