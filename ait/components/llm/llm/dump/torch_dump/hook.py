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
        self.tensor_ori_attr = {}
        self.torch_ori_attr = {}
        self.functional_ori_attr = {}
        self.distributed_ori_attr = {}
        self.npu_distributed_ori_attr = {}
        self.vf_ori_attr = {}
        self.aten_ori_attr = {}
        self.torch_npu_ori_attr = {}

        self.tensor_hook_attr = {}
        self.torch_hook_attr = {}
        self.functional_hook_attr = {}
        self.distributed_hook_attr = {}
        self.npu_distributed_hook_attr = {}
        self.vf_hook_attr = {}
        self.aten_hook_attr = {}
        self.torch_npu_hook_attr = {}
        self.hook_torch_ops = get_torch_ops()
        self.hook_function_ops = get_functional_ops()
        self.ori_module_attr = {}
        self.dump_config = DumpConfig()

    def store_ori_attr(self, module_name, api_list):
        for api_name in api_list:
            self.ori_module_attr[api_name] = getattr(module_name, api_name)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for api, api_attr in attr_dict.items():
            setattr(api_group, api, api_attr)

    def api_modularity(self):
        self.set_api_attr(torch.Tensor, self.tensor_hook_attr)
        self.set_api_attr(torch, self.torch_hook_attr)
        self.set_api_attr(torch.nn.functional, self.functional_hook_attr)
        if not is_gpu:
            self.set_api_attr(torch_npu, self.torch_npu_hook_attr)

    def api_originality(self):
        self.set_api_attr(torch.Tensor, self.tensor_ori_attr)
        self.set_api_attr(torch, self.torch_ori_attr)
        self.set_api_attr(torch.nn.functional, self.functional_ori_attr)

    def add_hook(self):
        self._add_module_hook()
        self._add_api_hook()

    def reset_hook(self):
        pass

    def _add_module_hook(self):
        model_name = "root"

        def add_hook(module, prefix=""):
            module.register_forward_hook(dump_module_hook())
            module.name = prefix
            for name, child_module in module.named_children():
                child_module.register_forward_hook(dump_module_hook())
                child_module.name = prefix + "." + name

            return
        # 遗留保存module的顺序
        self.model.name = model_name
        self.model.register_forward_pre_hook(set_dump_flag())
        self.model.register_forward_hook(dump_module_hook())
        for name_, module_ in self.model.named_children():
            add_hook(module_, prefix=model_name+"."+name_)

    def _add_api_hook(self):
        self.store_ori_attr(torch, self.hook_torch_ops)
        for ops in self.hook_torch_ops:
            wrap_func(ops)

        self.store_ori_attr(torch.nn.functional, self.hook_function_ops)
        for ops in self.hook_function_ops:
            wrap_func(ops)


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


