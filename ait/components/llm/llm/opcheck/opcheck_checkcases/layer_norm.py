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
from ait_llm.common.log import logger


class OpcheckLayerNormOperation(operation_test.OperationTest):
    def layer_norm_quant(self, layer_norm_res, quant_scale, quant_offset):
        golden_result_quant = (layer_norm_res * quant_scale + quant_offset).float()
        golden_result_quant = torch.round(golden_result_quant)
        golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
        return golden_result_quant.type(torch.int8)

    def golden_calc(self, in_tensors):
        layer_type = self.op_param['layerType']
        if layer_type == 1:
            cur_param = self.op_param['normParam']
        elif layer_type == 3:
            cur_param = self.op_param['postNormParam']
        else:
            raise ValueError('layerType should be 1 or 3')

        eps = cur_param['epsilon'] if 'epsilon' in cur_param.keys() else 1e-5
        is_quant = cur_param['quantType'] != 0 if "quantType" in cur_param.keys() else False
        quant_scale = cur_param['quantInputScale'] if 'quantInputScale' in cur_param.keys() else 1
        quant_offset = cur_param['quantInputOffset'] if 'quantInputOffset' in cur_param.keys() else 0
        quant_alpha = cur_param['quantInputAlpha'] if 'quantInputAlpha' in cur_param.keys() else 1

        if not is_quant:
            if layer_type == 1:
                op_input = in_tensors[0].float()
                weight = in_tensors[1].float()
                bias = in_tensors[2].float()
                axis = cur_param['beginNormAxis'] if 'beginNormAxis' in cur_param.keys() else 0
                normalized_shape = in_tensors[0].shape[axis:]
                golden_result = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
            elif layer_type == 3:
                weight = in_tensors[2].float()
                bias = in_tensors[3].float()
                normalized_shape = (1, in_tensors[0].shape[-1])
                zoom_scale_value = cur_param['zoomScale'] if 'zoomScale' in cur_param.keys() else 1
                op_input = torch.add(in_tensors[0].float(), zoom_scale_value * in_tensors[1].float())
                golden_result = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
            golden = [golden_result.half()] if in_tensors[0].dtype == torch.float16 else [golden_result]
        else:
            if layer_type == 1:
                op_input = in_tensors[0].float()
                weight = in_tensors[1].float()
                bias = in_tensors[2].float()                    
                normalized_shape = (1, in_tensors[0].shape[-1])
                layer_norm_res = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
                layer_norm_res = layer_norm_res.to(torch.float16)
                golden_result = (layer_norm_res * quant_alpha).to(torch.float16)
                golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_scale, quant_offset)
            elif layer_type == 3:
                weight = in_tensors[2].float()
                bias = in_tensors[3].float()
                normalized_shape = (1, in_tensors[0].shape[-1])                
                op_input = torch.add(in_tensors[0].float(), in_tensors[1].float())
                layer_norm_res = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
                layer_norm_res = layer_norm_res.to(torch.float16)
                golden_result = (layer_norm_res * quant_alpha).to(torch.float16)
                golden_result_quant = self.layer_norm_quant(layer_norm_res, quant_scale, quant_offset)
            golden = [golden_result, golden_result_quant]        

        return golden

    def test(self):
        self.execute()