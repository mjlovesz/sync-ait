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


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors, _):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        try:
            float_result = 0.5 * float_in_tensors * (1 + torch.nn.functional.tanh(np.sqrt(2 / np.pi) * 
                                                    (float_in_tensors + 0.044715 * torch.pow(float_in_tensors, 3))))
        except ZeroDivisionError as e:
            raise e       
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result

    @staticmethod
    def fast_gelu_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        try:
            float_result = float_in_tensors * torch.exp(0.851 * (float_in_tensors - torch.abs(float_in_tensors))) / \
                            (1 + torch.exp(-1.702 * torch.abs(float_in_tensors)))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def swish_golden(in_tensors, scale):
        float_in_tensors = in_tensors.float()
        try:
            float_result = float_in_tensors / (1 + torch.exp(-float_in_tensors * scale))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def log_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        float_result = torch.log(float_in_tensors)
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result


class OpcheckLinearActivationOperation(operation_test.OperationTest):
    def golden_flp(self, transpose_a: bool, transpose_b: bool, in_tensor_0, in_tensor_1):
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
        golden_result = np.matmul(in_tensor_0.astype(np.float32), in_tensor_1.astype(np.float32))
        golden_result = golden_result.astype(np.float16)
        return golden_result

    def golden_calc(self, in_tensors):
        golden_func = {
            1: ActivationGolden.relu_golden,
            2: ActivationGolden.gelu_golden,
            3: ActivationGolden.fast_gelu_golden,
            4: ActivationGolden.swish_golden,
            5: ActivationGolden.log_golden
        }

        transpose_a = self.op_param["transposeA"]
        transpose_b = not self.op_param["transposeB"]
        in_tensors[0] = in_tensors[0].cpu().numpy()
        in_tensors[1] = in_tensors[1].cpu().numpy()
        golden_result = self.golden_flp(transpose_a, transpose_b, in_tensors[0], in_tensors[1])
        if self.op_param["hasBias"]:
            in_tensors[2] = in_tensors[2].cpu().numpy()
            golden_result = golden_result.astype(np.float32) + in_tensors[2].astype(np.float32)
        golden_result = golden_result.astype(np.float16)
        golden_result = torch.tensor(golden_result).half()
        golden_result = self.golden_func[self.op_param["activationFuncType"]](golden_result, 0)
        return [golden_result.npu()]

    def test(self):
        self.execute()