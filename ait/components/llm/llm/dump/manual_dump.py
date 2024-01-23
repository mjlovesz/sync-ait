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
from llm.common.log import logger

def dump_data(token_id=-1, data_id=-1, golden_data=None, my_path='', output_path='./'):
    if token_id == -1 or data_id == -1 or golden_data is None or my_path == '':
        logger.warning('Please check whether the parameters passed in are correct')
        return

    if golden_data is not isinstance(golden_data, torch.Tensor):
        logger.warning('The golden_data is not a torch tensor!')
        return
    
    golden_data_dir = os.path.join(output_path, "golden_tensor", str(token_id))
    if not os.path.exists(golden_data_dir):
        os.makedirs(golden_data_dir)
    if golden_data is not None:
        golden_data_path = os.path.join(golden_data_dir, f'{data_id}_tensor.npy')
        golden_data = golden_data.cpu().numpy()
        np.save(golden_data_path, golden_data)

    json_path = os.path.join(output_path, "golden_tensor", "metadata.json")
    write_json_file(data_id, golden_data_path, json_path, token_id, my_path)
            

def write_json_file(data_id, data_path, json_path, token_id, my_path):
    # 建议与json解耦，需要的时候用
    import json

    try:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
    except FileNotFoundError:
        json_data = {}
    json_data[data_id] = {token_id: [data_path, my_path]}
    with ms_open(json_path, "w") as f:
        json.dump(json_data, f)