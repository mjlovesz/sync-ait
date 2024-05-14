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
import torch
import torch_npu

from ait_llm.opcheck import operation_test
from ait_llm.common.tool import read_atb_data
from ait_llm.common.log import logger


class OpcheckAllGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        def get_tensor_path(tensor_type):
            _tensor_path = [x for x in os.listdir(self.tensor_path) if x.startswith(tensor_type)]
            _tensor_path.sort(key=lambda x:int(x.split(tensor_type)[1].split('.')[0]))  
            _tensor_path = [os.path.join(self.tensor_path, x) for x in _tensor_path]
            return _tensor_path
 
        rank = self.op_param.get("rank", None) 
        rank_root = self.op_param.get("rankRoot", None)
        rank_size = self.op_param.get("rankSize", None)
        new_in_tensors = []
        for i in range(rank_root, rank_size):
            old_did_pid = "_".join([str(rank), str(self.pid)])
            new_did_pid = "_".join([str(i), str(int(self.pid) - int(rank) + i)])
            new_tensor_path = self.tensor_path.replace(old_did_pid, new_did_pid)
            if new_tensor_path:
                if os.path.isdir(new_tensor_path):
                    _in_tensor_path = get_tensor_path("intensor")
                    for path in _in_tensor_path:
                        _in_tensor = read_atb_data(path).npu()
                        new_in_tensors.append(_in_tensor) 
                else:
                    raise RuntimeError(f"{new_tensor_path} not valid")
            else:
                raise RuntimeError(f"{new_tensor_path} not valid")

        golden_result = torch.stack(new_in_tensors, dim=0)
        return [golden_result]        
  
    def test_all_gather(self):
        if self.pid is None:
            logger_text = f"Cannot get a valid pid, AllGatherOperation is not supported!"
            logger.error(logger_text)
            return

        rank = self.op_param.get("rank", None)
        rank_root = self.op_param.get("rankRoot", None)
        rank_size = self.op_param.get("rankSize", None)

        if rank is None or rank_root is None or rank_size is None:
            msg = "Cannot get golden data because opParam is not correctly set!"
            logger.error(msg)
            return
                 
        self.execute()