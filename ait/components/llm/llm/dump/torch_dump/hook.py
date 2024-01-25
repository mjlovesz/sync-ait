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

from llm.dump.torch_dump.dump import DumpConfig, dump_tensor, dump_module_hook, set_dump_flag
from llm.dump.torch_dump.hook_ops import HOOK_OPS


class HookModule:
    def __init__(self, model, dump_config=None):
        self.model = model
        self.dump_config = dump_config
        self.ori_torch_attr = {}

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
                add_hook(child_module, prefix + "." + name)

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
        for py_module, api_list in HOOK_OPS.items():
            ori_module_attrs = {}
            for api_name in api_list:
                if not hasattr(py_module, api_name):
                    continue
                api = getattr(py_module, api_name)
                ori_module_attrs[api_name] = api_name
                new_api = wrap_func(api)
                setattr(py_module, api_name, new_api)  # hook api

            self.ori_torch_attr[py_module] = ori_module_attrs

    def _remove_api_hook(self):
        for py_module, ori_attrs in self.ori_torch_attr:
            for api_name, api in ori_attrs:
                if not hasattr(py_module, api_name):
                    continue
                setattr(py_module, api_name, api)


def wrap_func(func):
    forward_count = 0

    @functools.wraps(func)
    def run(*args, **kwargs):
        nonlocal forward_count
        output = func(*args, **kwargs)
        dump_config = DumpConfig()
        if dump_config == "module":
            return output

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


def register_hook(model, dump_path="./", token_range=None, mode=None, module_list=None):
    dump_config = DumpConfig(dump_path=dump_path, token_range=token_range, mode=mode, module_list=module_list)
    hook_module = HookModule(model, dump_config)
    hook_module.add_hook()



