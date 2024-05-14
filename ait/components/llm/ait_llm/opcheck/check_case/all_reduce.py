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


class OpcheckAllReduceOperation(operation_test.OperationTest):
    def lccl_sum_cal(self, in_tensors):
        cal_tensors = [0] * 8 
        for idx, in_tensor in enumerate(in_tensors):
            cal_tensors[idx] = in_tensor
        result = ((cal_tensors[0] + cal_tensors[1]) + (cal_tensors[2] + cal_tensors[3])) + ((cal_tensors[4] + 
                cal_tensors[5]) + (cal_tensors[6] + cal_tensors[7]))
        return [result]

    def sum_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)):
            result += in_tensors[i]
        return [result]

    def max_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)): 
            result = torch.max(result, in_tensors[i])
        return [result]

    def min_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)): 
            result = torch.min(result, in_tensors[i])
        return [result]

    def prod_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)):
            result = torch.mul(result, in_tensors[i])
        return [result]

    def get_new_in_tensors(self):
        def get_tensor_path(new_tensor_path, tensor_type):
            _tensor_path = [x for x in os.listdir(new_tensor_path) if x.startswith(tensor_type)]
            _tensor_path.sort(key=lambda x:int(x.split(tensor_type)[1].split('.')[0]))  
            _tensor_path = [os.path.join(new_tensor_path, x) for x in _tensor_path]
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
                    _in_tensor_path = get_tensor_path(new_tensor_path, "intensor")
                    for path in _in_tensor_path:
                        _in_tensor = read_atb_data(path).npu()
                        new_in_tensors.append(_in_tensor) 
                else:
                    raise RuntimeError(f"{new_tensor_path} not valid")
            else:
                raise RuntimeError(f"{new_tensor_path} not valid")
        
        return new_in_tensors

    def golden_calc(self, in_tensors):
        all_reduce_type = self.op_param.get('allReduceType', None)
        backend = self.op_param.get('backend', None)

        new_in_tensors = self.get_new_in_tensors()
                    
        if all_reduce_type == "sum":
            if backend == "lccl":
                golden = self.lccl_sum_cal(new_in_tensors)
            else:
                golden = self.sum_cal(new_in_tensors)
        elif all_reduce_type == "max":
            golden = self.max_cal(new_in_tensors)
        elif all_reduce_type == "min":
            golden = self.min_cal(new_in_tensors)
        elif all_reduce_type == "prod":
            golden = self.prod_cal(new_in_tensors)

        return golden

    def test_all_reduce(self):
        all_reduce_type = self.op_param.get('allReduceType', None)
        backend = self.op_param.get('backend', None)

        logger_text1 = f"backend: {backend}, allreduceType: {all_reduce_type}"
        logger_text2 = "env: {}".format(os.getenv("LCCL_DETERMINISTIC", ""))
        logger_text3 = "env: {}".format(os.getenv("HCCL_DETERMINISTIC", ""))
        logger.debug(logger_text1)
        logger.debug(logger_text2)
        logger.debug(logger_text3)

        if all_reduce_type is None or backend is None:
            msg = "Cannot get golden data because opParam is not correctly set!"
            logger.error(msg)
            return
        self.execute()