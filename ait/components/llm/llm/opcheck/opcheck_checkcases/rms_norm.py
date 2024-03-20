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


class OpcheckRmsNormOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        layertype = self.op_param['layerType']
        if layertype == 1:
            cur_param = self.op_param['normParam']
        elif layertype == 2:
            cur_param = self.op_param['preNormParam']
        else:
            raise ValueError('layerType should be 1 or 2')
        
        quant_type = cur_param['quantType']
        eps = cur_param['epsilon'] if 'epsilon' in cur_param.keys() else 0.00001
        x = in_tensors[0].float()
        gamma = in_tensors[1].float()
        gamma = gamma.view(1, -1)
        if layertype == 2 and quant_type == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        gamma_size = float(gamma.size(-1))
        try:
            norm = torch.sum(x / gamma_size * x, dim=-1, keepdim=True) + eps
            golden_output = x * gamma / torch.sqrt(norm)
        except ZeroDivisionError as e:
            raise e

        def rms_norm_quant(golden_output, beta):
            golden_output = golden_output.float()
            beta = beta.float()
            quant_scale = cur_param['quantInputScale'] if 'quantInputScale' in cur_param.keys() else 1
            quant_offset = cur_param['quantInputOffset'] if 'quantInputOffset' in cur_param.keys() else 0
            golden_output = golden_output + beta
            golden_output = golden_output * quant_scale + quant_offset
            golden_output = torch.clamp(golden_output, -128, 127)
            golden_result_quant = torch.round(golden_output)
            return golden_result_quant.type(torch.int8)
    
        if layertype == 2 and quant_type == 2:
            golden_result = [x, rms_norm_quant(golden_output, in_tensors[3])]
        elif layertype == 1 and quant_type == 2:
            golden_result = [rms_norm_quant(golden_output, in_tensors[2])]
        else:
            golden_result = [golden_output.half()]

        return golden_result

    def test(self):
        self.execute()

