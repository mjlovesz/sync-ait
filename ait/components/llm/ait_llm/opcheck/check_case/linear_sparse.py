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
import sys
import json
import unittest
import torch
import torch_npu
import numpy as np

from ait_llm.opcheck import operation_test


class OpcheckLinearSparseOperation(operation_test.OperationTest):
    def golden_quant(self, transpose_a: bool, transpose_b: bool, in_tensors):
        in_tensor_0 = in_tensors[0]
        in_tensor_1 = in_tensors[1]
        in_tensor_2 = in_tensors[2] 
        in_tensor_3 = in_tensors[3]
        if transpose_a:
            if len(in_tensor_0.shape) == 2:
                in_tensor_0 = np.transpose(in_tensor_0, (1, 0))
            if len(in_tensor_0.shape) == 3:
                in_tensor_0 = np.transpose(in_tensor_0, (0, 2, 1))
            in_tensor_0 = np.ascontiguousarray(in_tensor_0)
        if len(in_tensor_1.shape) == 4:
            in_tensor_1 = np.transpose(in_tensor_1, (0, 2, 1, 3))
            if in_tensor_1.shape[0] == 1:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[1], in_tensor_1.shape[2] * in_tensor_1.shape[3])
            else:
                in_tensor_1 = in_tensor_1.reshape(in_tensor_1.shape[0], in_tensor_1.shape[1],
                                                  in_tensor_1.shape[2] * in_tensor_1.shape[3])
            in_tensor_1 = np.ascontiguousarray(in_tensor_1)
        if transpose_b:
            if len(in_tensor_1.shape) == 2:
                in_tensor_1 = np.transpose(in_tensor_1, (1, 0))
            if len(in_tensor_1.shape) == 3:
                in_tensor_1 = np.transpose(in_tensor_1, (0, 2, 1))
            in_tensor_1 = np.ascontiguousarray(in_tensor_1)
        golden_result = np.matmul(in_tensor_0.astype(np.int32), in_tensor_1.astype(np.int32))
        golden_result = golden_result + in_tensor_2
        golden_result = golden_result * in_tensor_3
        golden_result = golden_result.astype(np.float16)
        return golden_result

    def golden_calc(self, in_tensors):
        transpose_a = self.op_param["transposeA"]
        transpose_b = self.op_param["transposeB"]
        in_tensors[0] = in_tensors[0].cpu().numpy()
        in_tensors[1] = in_tensors[1].cpu().numpy()
        in_tensors[2] = in_tensors[2].cpu().numpy()
        in_tensors[3] = in_tensors[3].cpu().numpy()
        golden_result = self.golden_quant(transpose_a, transpose_b, in_tensors)
        golden_result = torch.tensor(golden_result).half()
        return [golden_result.npu()]

    def test(self):
        self.excute()