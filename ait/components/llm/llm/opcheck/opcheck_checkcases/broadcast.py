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
import json
import unittest
import sys
import socket
import random
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from ait_llm.opcheck import operation_test


class OpcheckBroadcastOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        rank_root = self.op_param['rankRoot']
        golden_result = in_tensors[rank_root]
        return [golden_result]

    def test_broadcast(self):
        self.execute()