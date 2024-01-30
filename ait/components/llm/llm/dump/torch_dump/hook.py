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
import torch

from llm.common.log import logger
from llm.dump.torch_dump.dump_config import DumpConfig
from llm.dump.torch_dump.dump_hook import DumpHookModule


def register_hook(model, hook_type="dump_data", config=None):
    if not isinstance(model, torch.nn.Module):
        logger.error("model must be instance of torch.nn.Module.")
    if hook_type != "dump_data":
        logger.error("hook_type must be dump_data.")
    if not isinstance(config, DumpConfig):
        logger.error("model must be instance of DumpConfig.")

    dump_config = config or DumpConfig()
    if hook_type == "dump_data":
        hook_module = DumpHookModule(model, dump_config)
        hook_module.add_hook()
