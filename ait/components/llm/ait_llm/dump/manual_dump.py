# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import numpy as np
import torch

from components.utils.file_open_check import ms_open
from ait_llm.common.log import logger


def dump_data(token_id=-1, data_id=-1, golden_data=None, my_path='', output_path='./'):
    # 传参失败的提示
    if token_id == -1:
        logger.warning('Please check whether token_id passed in are correct')
        return
    elif data_id == -1:
        logger.warning('Please check whether data_id passed in are correct')
        return
    elif not isinstance(golden_data, torch.Tensor):
        logger.warning('The golden_data is not a torch tensor!')
        return
    elif my_path == '':
        logger.warning('Please check whether my_path passed in are correct')
        return

    if golden_data is not None:
        cur_pid = os.getpid()
        device_id = golden_data.get_device()
        golden_data_dir = os.path.join(output_path, "ait_dump", f"{cur_pid}_{device_id}", "golden_tensor",
                                       str(token_id))

        if not os.path.exists(golden_data_dir):
            os.makedirs(golden_data_dir)

        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.pth')
        torch.save(golden_data, golden_data_path)

        json_path = os.path.join(output_path, "ait_dump", f"{cur_pid}_{device_id}", "golden_tensor", "metadata.json")
        write_json_file(data_id, golden_data_path, json_path, token_id, my_path)


def write_json_file(data_id, data_path, json_path, token_id, my_path):
    import json
    if not os.path.exists(json_path):
        json_data = {}
    else:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

    json_data[data_id] = {token_id: [data_path, my_path]}
    with ms_open(json_path, "w") as f:
        json.dump(json_data, f)