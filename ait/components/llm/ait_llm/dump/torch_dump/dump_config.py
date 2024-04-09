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

from ait_llm.common import utils
from ait_llm.common.log import logger


def singleton(cls):
    ins = {}

    def run(*args, **kwargs):
        if cls not in ins:
            ins[cls] = cls(*args, **kwargs)
        return ins.get(cls)

    return run


@singleton
class DumpConfig:
    def __init__(self, dump_path=None, token_range=None, module_list=None, tensor_part=2, device_id=None):
        self.dump_path = dump_path or "./"
        self.mode = "module"
        self.token_range = token_range or [0]
        self.module_list = module_list or []
        self.tensor_part = tensor_part
        self.device_id = device_id
        self.is_dump_cur_device = True
        self.dump_flag = True
        self.token_id = 0
        self.module_ids = {}
        self.cur_module_id = 0
        self.dump_dir = ""

        if not self._check_args():
            raise ValueError("Invalid args of DumpConfig.")

    def set_dump_device_and_dump_dir(self, device):
        if self.device_id is not None and device != "cpu":
            # Get the first position of a digit char, and cut out like cuda0 -> cuda, npu12 -> npu
            device_type = device[:max(enumerate(device), key=lambda xx: str.isdigit(xx[1]))[0]]
            device_id_str = f"{device_type}{self.device_id}"  # -> npu0
            if device != device_id_str:
                self.is_dump_cur_device = False
                return

        cur_dump_path = "{}_{}".format(device, os.getpid())
        self.dump_dir = os.path.join(self.dump_path, "ait_dump", "torch_tensors", cur_dump_path)
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, mode=0o750)

    def update_module_ids(self, module_name):
        self.cur_module_id += 1
        if module_name not in self.module_ids:
            self.module_ids[module_name] = self.cur_module_id

    def _check_args(self):
        utils.check_output_path_legality(self.dump_path)
        if not isinstance(self.token_range, list):
            logger.error("dump_path must be list.")
            return False
        if not isinstance(self.module_list, list):
            logger.error("module_list must be list.")
            return False
        if not isinstance(self.tensor_part, int):
            logger.error("tensor_part must be int.")
            return False
        if self.device_id is not None and not isinstance(self.device_id, int):
            logger.error("device_id must be int.")
            return False
        if self.tensor_part not in [0, 1, 2]:
            logger.error("tensor_part must be 0 or 1 or 2.")
            return False
        return True
