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

from llm.common.log import logger


def try_import_torchair():
    try:
        import torch
        import torch_npu
        import torchair
    except ModuleNotFoundError as ee:
        logger.error("torch or torch_npu with torchair not found. Try installing the latest torch_npu.")
        raise ee


def get_ge_dump_config(dump_path="ait_ge_dump", dump_mode="all", use_fusion=True):
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    if not os.path.exists(dump_path):
        os.makedirs(dump_path, mode=0o750)

    # Generate GE mapping graph
    config.debug.graph_dump.type = "txt"
    config.debug.graph_dump.path = dump_path

    # Enable GE dump
    config.dump_config.enable_dump = True
    config.dump_config.dump_mode = dump_mode
    config.dump_config.dump_path = dump_path

    return config


def get_fx_dump_config(dump_path="ait_ge_dump", dump_mode="all"):
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    # Enable FX dump
    config.debug.data_dump.type = "npy"
    return config
