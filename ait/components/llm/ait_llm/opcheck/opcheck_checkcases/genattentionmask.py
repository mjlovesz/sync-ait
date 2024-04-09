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


class OpcheckElewiseSubOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        out = []
        for i, s in enumerate(self.op_param['seqLen']):
            for _ in range(self.op_param["headNum"]):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def test_2d_half(self):
        self.execute()