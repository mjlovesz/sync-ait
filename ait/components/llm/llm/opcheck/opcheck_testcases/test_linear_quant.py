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

from llm.opcheck import operation_test


class TestLinearQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensors[0].to(torch.int32), in_tensors[1].to(torch.int32))
        if self.op_param["hasBias"]:
            golden_result = golden_result + in_tensors[2]
        golden_result = golden_result * in_tensors[3]
        golden_result = golden_result.to(torch.float16)
        return [golden_result.npu()]

    def test(self):
        self.execute()