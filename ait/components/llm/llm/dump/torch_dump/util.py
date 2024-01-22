#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
import torch

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

torch_without_guard_version_list = ['2.1']
for version in torch_without_guard_version_list:
    if torch.__version__.startswith(version):
        torch_without_guard_version = True
        break
    else:
        torch_without_guard_version = False


def get_torch_ops():
    global WrapTorchOps
    _torch_ops = dir(torch._C._VariableFunctionsClass)
    return set(WrapTorchOps) & set(_torch_ops)


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    return set(WrapFunctionalOps) & set(_all_functional_ops)
