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
import functools
import os
import os.path

import numpy as np
import torch

from ait_llm.dump.torch_dump.dump_config import DumpConfig
from ait_llm.dump.torch_dump import hook_ops
from ait_llm.common.log import logger


class DumpHookModule:
    def __init__(self, model, dump_config=None):
        self.model = model
        self.dump_config = dump_config
        self.ori_torch_attr = {}
        hook_ops.add_torch_ops()
        hook_ops.add_torch_npu_ops()

    def add_hook(self):
        self._add_module_hook()
        self._add_api_hook()

    def remove_hook(self):
        self._remove_module_hook()
        self._remove_api_hook()

    def _add_module_hook(self):
        model_name = "root"

        def add_hook(module, prefix=""):
            module.ait_forward_handle = module.register_forward_hook(dump_module_data())
            module.name = prefix
            for name, child_module in module.named_children():
                add_hook(child_module, prefix + "." + name)

        self.model.name = model_name
        self.model.ait_forward_pre_handle = self.model.register_forward_pre_hook(set_dump_flag())
        add_hook(self.model, prefix=model_name)

    def _remove_module_hook(self):
        if hasattr(self.model, "ait_forward_pre_handle"):
            self.model.ait_forward_pre_handle.remove()

        def _remove_hook(module):
            if hasattr(module, "ait_forward_handle"):
                module.ait_forward_handle.remove()
            for _, _child_module in module.named_children():
                _remove_hook(_child_module)

        _remove_hook(self.model)

    def _add_api_hook(self):
        for py_module, api_list in hook_ops.HOOK_OPS.items():
            ori_module_attrs = {}
            for api_name in api_list:
                if not hasattr(py_module, api_name):
                    continue
                api = getattr(py_module, api_name)
                ori_module_attrs[api_name] = api_name
                new_api = wrap_torch_func(api)
                setattr(py_module, api_name, new_api)  # hook api

            self.ori_torch_attr[py_module] = ori_module_attrs

    def _remove_api_hook(self):
        for py_module, ori_attrs in self.ori_torch_attr:
            for api_name, api in ori_attrs:
                if not hasattr(py_module, api_name):
                    continue
                setattr(py_module, api_name, api)


def wrap_torch_func(func):
    exec_count = 0

    @functools.wraps(func)
    def dump_api_data(*args, **kwargs):
        nonlocal exec_count
        exec_count += 1
        output = func(*args, **kwargs)

        dump_config = DumpConfig()
        if not dump_config.dump_flag or not dump_config.is_dump_cur_device or dump_config.mode == "module":
            return output

        api_dump_path = os.path.join(dump_config.dump_dir, func.__name__, str(exec_count))
        if not os.path.exists(api_dump_path):
            os.makedirs(api_dump_path)
        dump_data(args, output, api_dump_path, dump_config.tensor_part)
        return output

    return dump_api_data


def dump_tensor(feat, feat_path):
    if isinstance(feat, (tuple, list)):
        for idx, tensor in enumerate(feat):
            dump_tensor(tensor, "{}_{}".format(feat_path, idx))
    elif isinstance(feat, torch.Tensor):
        if not feat_path.endswith(".pth"):
            feat_path += ".pth"
        torch.save(feat, feat_path)
    else:
        logger.warning("Unrecognized data type %s, cannot be saved in path %s.", type(feat), feat_path)


def dump_data(inputs, outputs, dump_path, tensor_part):
    if tensor_part == 0:
        dump_tensor(inputs, os.path.join(dump_path, "input"))
    elif tensor_part == 1:
        dump_tensor(outputs, os.path.join(dump_path, "output"))
    else:
        dump_tensor(inputs, os.path.join(dump_path, "input"))
        dump_tensor(outputs, os.path.join(dump_path, "output"))


def dump_module_data():
    exec_count = 0

    def hook_func(module: torch.nn.Module, inputs, outputs):
        nonlocal exec_count
        exec_count += 1
        dump_config = DumpConfig()

        if not dump_config.is_dump_cur_device:
            return
        
        if dump_config.token_id == 0:
            dump_config.update_module_ids(module.name)
            # 将模型树状信息保存成json文件
            from ait_llm.dump.torch_dump.topo import ModelTree
            if not os.path.exists(dump_config.dump_dir):
                os.makedirs(dump_config.dump_dir, mode=0o750)
            model_tree_path = os.path.join(dump_config.dump_dir, "model_tree.json")
            obj = ModelTree()
            obj.create_tree(module, dump_config.module_ids, model_tree_path)

        if dump_config.mode == "api" or not dump_config.dump_flag:
            return

        if dump_config.module_list and not isinstance(module, tuple(dump_config.module_list)):
            return

        module_name = module.name
        dump_path = os.path.join(dump_config.dump_dir, str(dump_config.token_id), module_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path, mode=0o750)
        dump_data(inputs, outputs, dump_path, dump_config.tensor_part)

    return hook_func


def set_dump_flag():
    cur_token_id = 0

    def hook_func(module, inputs):
        nonlocal cur_token_id
        logger.debug("Current token id: %s", cur_token_id)
        config = DumpConfig()
        config.token_id = cur_token_id
        # 通过root module执行的轮次来判断当前在第几个token
        if config.token_id in config.token_range:
            config.dump_flag = True
        else:
            config.dump_flag = False

        cur_token_id += 1

    return hook_func
