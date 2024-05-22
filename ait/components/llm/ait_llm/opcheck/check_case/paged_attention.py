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

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class OpcheckPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if 'isSupportAlibi' in self.op_param:
            is_support_alibi = self.op_param["isSupportAlibi"]
        else:
            is_support_alibi = False
        
        if is_support_alibi:
            return [in_tensors[6]]
        else:
            return [in_tensors[5]]

    def test(self):
        logger_text = f"PagedAttentionOperation is not supported!"
        logger.error(logger_text)
        return