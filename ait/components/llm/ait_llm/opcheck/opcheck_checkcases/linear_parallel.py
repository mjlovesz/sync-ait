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
import unittest
import torch
import torch_npu
import numpy as np
import torch.distributed as dist

from ait_llm.opcheck import operation_test


class OpcheckLinearParallelOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensor_0.to(torch.float32), in_tensor_1.to(torch.float32)).to(torch.float16)
        dist.all_reduce(golden_result, op=ReduceOp.SUM)
        torch.npu.synchronize()
        if in_tensors[2] is not None:
            golden_result = golden_result + in_tensors[2]

        return [golden_result]

    def test(self):
        self.execute()