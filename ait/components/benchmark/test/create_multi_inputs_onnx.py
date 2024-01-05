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

import torch
from torch import nn


class AA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(3, 32, 3, 2, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 16 * 16, 10)

    def forward(self, input_1, input_2):
        out_1 = self.conv1(input_1)
        out_2 = self.conv2(input_2)
        out = out_1 + out_2

        out = self.flatten(out)
        out = self.linear(out)
        return out


aa = AA()
aa(torch.ones([1, 3, 32, 32]), torch.zeros([1, 3, 32, 32])).shape
torch.onnx.export(
    aa, (torch.ones([1, 3, 32, 32]), torch.zeros([1, 3, 32, 32])), 'multi_dym_aipp_model.onnx', opset_version=11
)
