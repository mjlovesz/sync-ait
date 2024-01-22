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
import os.path

import torch
try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

from llm.dump.torch_dump.util import get_torch_ops, get_functional_ops
from llm.dump.torch_dump.dump import DumpConfig, dump_tensor, dump_module_hook, set_dump_flag


class HookModule:
    def __init__(self, model):
        self.model = model
        self.hook_torch_ops = get_torch_ops()
        self.hook_function_ops = get_functional_ops()
        self.ori_torch_ops_attr = {}
        self.ori_function_ops_attr = {}

    def add_hook(self):
        self._add_module_hook()
        self._add_api_hook()

    def remove_hook(self):
        self._remove_module_hook()
        self._remove_api_hook()

    def _add_module_hook(self):
        model_name = "root"

        def add_hook(module, prefix=""):
            module.ait_forward_handle = module.register_forward_hook(dump_module_hook())
            module.name = prefix
            for name, child_module in module.named_children():
                module.ait_forward_handle = child_module.register_forward_hook(dump_module_hook())
                child_module.name = prefix + "." + name

        self.model.name = model_name
        self.model.ait_forward_pre_handle = self.model.register_forward_pre_hook(set_dump_flag())
        add_hook(self.model, prefix=model_name)

    def _remove_module_hook(self):
        self.model.ait_forward_pre_handle.remove()
        self.model.ait_forward_handle.remove()

        def _remove_hook(module):
            module.ait_forward_handle.remove()
            for _, _child_module in module.named_children():
                _remove_hook(_child_module)

        _remove_hook(self.model)

    def _add_api_hook(self):
        for ops in self.hook_torch_ops:
            self.ori_torch_ops_attr[ops] = getattr(torch, ops)
            new_ops = wrap_func(ops)
            setattr(torch, ops, new_ops)
            # self.new_torch_ops_attr[ops] = new_ops

        for ops in self.hook_function_ops:
            self.ori_function_ops_attr[ops] = getattr(torch.nn.functional, ops)
            new_ops = wrap_func(ops)
            setattr(torch.nn.functional, ops, new_ops)
            # self.new_function_ops_attr[ops] = new_ops

    def _remove_api_hook(self):
        for ops in self.hook_torch_ops:
            setattr(torch, self.ori_torch_ops_attr.get(ops))

        for ops in self.hook_function_ops:
            setattr(torch.nn.functional, self.ori_function_ops_attr.get(ops))


def wrap_func(func):
    forward_count = 0

    @functools.wraps
    def run(*args, **kwargs):
        nonlocal forward_count
        output = func(*args, **kwargs)
        dump_config = DumpConfig()
        api_dump_path = os.path.join(dump_config.dump_path, func.__name__, str(forward_count))
        if not os.path.exists(api_dump_path):
            os.makedirs(api_dump_path)
        if dump_config.tensor_part == "1":
            dump_tensor(output, os.path.join(api_dump_path, "output"))
        elif dump_config.tensor_part == "0":
            dump_tensor(args, os.path.join(api_dump_path, "input"))
        else:
            dump_tensor(args, os.path.join(api_dump_path, "input"))
            dump_tensor(output, os.path.join(api_dump_path, "output"))

        return output

    return run


