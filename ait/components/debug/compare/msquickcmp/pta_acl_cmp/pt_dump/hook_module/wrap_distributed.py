#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import os

import torch.distributed as dist
import yaml

from .hook_module import HOOKModule
from ..common.utils import torch_device_guard


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapDistributedOps = yaml.safe_load(f).get('distributed')


distributed_func = {}
for f in dir(dist):
    distributed_func[f] = getattr(dist, f)


def get_distributed_ops():
    global WrapDistributedOps
    _all_distributed_ops = dir(dist)
    return set(WrapDistributedOps) & set(_all_distributed_ops)


class HOOKDistributedOP(object):
    pass


class DistributedOPTemplate(HOOKModule):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Distributed_" + str(op_name) + "_"
        super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return distributed_func.get(self.op_name_)(*args, **kwargs)


def wrap_distributed_op(op_name, hook):
    def distributed_op_template(*args, **kwargs):
        return DistributedOPTemplate(op_name, hook)(*args, **kwargs)

    return distributed_op_template


def wrap_distributed_ops_and_bind(hook):
    _distributed_ops = get_distributed_ops()
    for op_name in _distributed_ops:
        setattr(HOOKDistributedOP, "wrap_" + str(op_name), wrap_distributed_op(op_name, hook))
