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

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckBroadcastOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        rank_root = self.op_param.get('rankRoot', None)
        golden_result = in_tensors[rank_root]
        return [golden_result]

    def test_broadcast(self):
        rank_root = self.op_param.get('rankRoot', None)
        if rank_root is None:
            msg = "Cannot get golden data because rankRoot is not correctly set!"
            logger.error(msg)
            return
        self.execute()