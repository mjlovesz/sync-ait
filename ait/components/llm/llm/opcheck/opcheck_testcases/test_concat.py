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

from llm.opcheck import operation_test


class TestConcatOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        concat_dim = self.op_param["concatDim"]
        axis_num = concat_dim if concat_dim >= 0 else concat_dim + len(in_tensors[0].size())
        golden_result = torch.cat(in_tensors, axis=axis_num)
        return [golden_result]

    def test(self):
        self.execute()
