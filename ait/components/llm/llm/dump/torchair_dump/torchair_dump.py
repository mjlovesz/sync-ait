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
from llm.common.log import logger

def try_import_torchair():
    try:
        import torch
        import torch_npu
        import torchair
    except:
        raise ModuleNotFoundError("torch or torch_npu with torchair not found. Try installing the latest torch_npu.")


def get_ge_dump_config(dump_path="ait_ge_dump", dump_mode="all", use_fusion=True):
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()

    # 打印映射关系
    config.debug.graph_dump.type = "txt"
    config.debug.graph_dump.path = dump_path

    # 是能 GE dump
    config.dump_config.enable_dump = True
    config.dump_config.dump_mode = dump_mode
    config.dump_config.dump_path = dump_path

    return config


def get_fx_dump_config(dump_path="ait_ge_dump", dump_mode="all"):
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.dump.data_dump.type = "npy"
    return config