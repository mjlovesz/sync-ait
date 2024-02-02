import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestLayerNormOperation(operation_test.OperationTest):
    def layer_norm_quant(self, layer_norm_res):
        golden_result_quant = (layer_norm_res * quant_scale + quant_offset).float()
        golden_result_quant = torch.round(golden_result_quant)
        golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
        return golden_result_quant.type(torch.int8)

    def golden_calc(self, in_tensors):
        eps = self.op_param['epsilon'] if 'epsilon' in self.op_param.keys() else 1e-5
        is_quant = self.op_param['quantType'] != 0
        quant_scale = self.op_param['quantInputScale'] if 'quantInputScale' in self.op_param.keys() else 1
        quant_offset = self.op_param['quantInputOffset'] if 'quantInputOffset' in self.op_param.keys() else 0
        quant_alpha = self.op_param['quantInputAlpha'] if 'quantInputAlpha' in self.op_param.keys() else 1
        layer_type = self.op_param['layerType']

        if not is_quant:
            if layer_type == 1:
                op_input = in_tensors[0].float()
                weight = in_tensors[1].float()
                bias = in_tensors[2].float()
                axis = self.op_param['beginNormAxis'] if 'beginNormAxis' in self.op_param.keys() else 0
                normalized_shape = in_tensors[0].shape[axis:]
                golden_result = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
            elif layer_type == 3:
                weight = in_tensors[2].float()
                bias = in_tensors[3].float()
                normalized_shape = (1, in_tensors[0].shape[-1])
                zoom_scale_value = self.op_param['zoomScaleValue']
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
                golden_result_quant = self.layer_norm_quant(layer_norm_res)
            elif layer_type == 3:
                weight = in_tensors[2].float()
                bias = in_tensors[3].float()
                normalized_shape = (1, in_tensors[0].shape[-1])                
                op_input = torch.add(in_tensors[0].float(), in_tensors[1].float())
                layer_norm_res = torch.nn.functional.layer_norm(op_input, normalized_shape, weight, bias, eps)
                layer_norm_res = layer_norm_res.to(torch.float16)
                golden_result = (layer_norm_res * quant_alpha).to(torch.float16)
                golden_result_quant = self.layer_norm_quant(layer_norm_res)
            golden = [golden_result, golden_result_quant]        

        return golden

    def test(self):
        self.execute()