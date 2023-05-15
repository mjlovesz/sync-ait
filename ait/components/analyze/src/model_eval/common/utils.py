# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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
import subprocess

from typing import List

from model_eval.common import logger
from model_eval.common.enum import Framework, SocType


def get_soc_type() -> str:
    default_soc = SocType.Ascend310.name
    try:
        import acl
        return acl.get_soc_name()
    except ImportError:
        logger.warning(
            f'Get soc failed, use default {default_soc}.'
        )
    return default_soc


def get_framework(model: str) -> Framework:
    dict_ = {
        # file format and framework map
        '.onnx': Framework.ONNX,
        '.prototxt': Framework.CAFFE,
        '.pb': Framework.TF
    }
    for suffix, framework in dict_.items():
        if model.endswith(suffix):
            return framework
    return Framework.UNKNOWN


def exec_command(cmd_args: List[str]):
    out = subprocess.run(cmd_args, capture_output=True, shell=False)
    outmsg = out.stdout.decode('utf-8')
    outerr = out.stderr.decode('utf-8')
    return outmsg, outerr


def check_file_security(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return True
    if os.path.islink(filepath):
        logger.error(f'{filepath} is link.')
        return False
    if not os.path.isfile(filepath):
        logger.error(f'{filepath} is not file.')
        return False
    return True
