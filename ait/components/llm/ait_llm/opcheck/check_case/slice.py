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


class OpcheckSliceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        offset_list = self.op_param.get('offsets', None)
        size_list = self.op_param.get('size', None)
        for index, offset in enumerate(offset_list):
            offset_list[index] = offset if offset >= 0 else offset + in_tensors[0].shape[index]
        for index, size in enumerate(size_list):
            size_list[index] = size if size != -1 else in_tensors[0].shape[index] - offset_list[index]
        if len(offset_list) == 3:
            return [in_tensors[0][offset_list[0] : offset_list[0] + size_list[0], 
                    offset_list[1] : offset_list[1] + size_list[1], offset_list[2] : offset_list[2] + size_list[2]]]
        else:
            return [in_tensors[0][offset_list[0] : offset_list[0] + size_list[0], 
                    offset_list[1] : offset_list[1] + size_list[1], offset_list[2] : offset_list[2] + size_list[2], 
                    offset_list[3] : offset_list[3] + size_list[3]]]

    def test(self):
        offset_list = self.op_param.get('offsets', None)
        size_list = self.op_param.get('size', None)
        if offset_list is None or size_list is None:
            msg = "Cannot get golden data because opParam is not correctly set!"
            logger.error(msg)
            return
        self.execute()