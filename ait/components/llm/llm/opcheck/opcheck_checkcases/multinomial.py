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


class OpcheckMultinomialOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        samples = self.op_param["numSamples"]
        rand_seed = self.op_param["randSeed"]
        input0 = in_tensors[0].cpu().numpy()
        libc = CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]
        ret = np.zeros(shape=(input0.shape[0], samples))

        sum_list = np.cumsum(input0, axis=-1, dtype=np.float16).astype(np.float16)
        iter_list = [(j, i) 
                    for j in range(input0.shape[0]) 
                    for i in range(input0.shape[1])]
        for z in range(samples):
            for j, i in iter_list:
                if (sum_list[j][i] > rand_list[z]):
                    ret[j][z] = i
                    break
        return [torch.from_numpy(ret.astype(np.int32)).contiguous()]

    def test(self):
        self.execute()