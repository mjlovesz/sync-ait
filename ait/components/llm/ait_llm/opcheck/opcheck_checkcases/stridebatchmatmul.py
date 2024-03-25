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

import sys
import os
import unittest
import torch
import torch_npu

from ait_llm.opcheck import operation_test


class OpcheckStridedBatchMatmulOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        a = in_tensors[0].flatten()
        b = in_tensors[1].flatten()

        batch_start_a = 0
        batch_start_b = 0
        batch_start_c = 0
        list_a = []
        list_b = []
        batch = self.op_param["batch"]
        head_num = self.op_param["headNum"]
        c = torch.zeros(sum([self.op_param["m"][i] * self.op_param["n"][i] for i in range(batch)]) * head_num, 
                        dtype=torch.float16, device=a.device)
        trans_a = self.op_param["transA"]
        trans_b = self.op_param["transB"]

        for i in range(batch):
            for j in range(head_num):
                list_a = []
                list_b = []
                row_a = self.op_param["m"][i] if not trans_a else self.op_param["k"][i]
                col_a = self.op_param["k"][i] if not trans_a else self.op_param["m"][i]
                for t in range(row_a):
                    start_a = self.op_param["lda"][i] * t + self.op_param["strideA"][i] * j + batch_start_a
                    end_a = start_a + col_a
                    list_a.append(a[start_a:end_a])
                row_b = self.op_param["k"][i] if not trans_b else self.op_param["n"][i]
                col_b = self.op_param["n"][i] if not trans_b else self.op_param["k"][i]
                for t in range(row_b):
                    start_b = self.op_param["ldb"][i] * t + self.op_param["strideB"][i] * j + batch_start_b
                    end_b = start_b + col_b
                    list_b.append(b[start_b:end_b])
                mat_a = torch.stack(list_a)
                mat_b = torch.stack(list_b)
                mat_a = torch.transpose(mat_a, 0, 1) if trans_a else mat_a
                mat_b = torch.transpose(mat_b, 0, 1) if trans_b else mat_b
                mat_c = torch.matmul(mat_a.float(), mat_b.float()).half()
                for t in range(mat_c.shape[0]):
                    start_c = self.op_param["ldc"][i] * t + self.op_param["strideC"][i] * j + batch_start_c
                    end_c = start_c + mat_c.shape[1]
                    c[start_c:end_c] = mat_c[t, :]
            batch_start_a += self.op_param["m"][i] * self.op_param["k"][i] * head_num
            batch_start_b += self.op_param["n"][i] * self.op_param["k"][i] * head_num
            batch_start_c += self.op_param["m"][i] * self.op_param["n"][i] * head_num
        return [c]

    def test_add_bmm1(self):
        self.execute()