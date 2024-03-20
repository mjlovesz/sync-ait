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

import json
import math
import os
import random
import sys
import unittest

import numpy as np
import torch
import torch_npu

from ait_llm.opcheck import operation_test


class OpcheckReshapeAndCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden = []
        for index in self.case_info['inplace_idx']:
            golden.append(in_tensors[index])
        return golden

    def test(self):
        self.execute_inplace()