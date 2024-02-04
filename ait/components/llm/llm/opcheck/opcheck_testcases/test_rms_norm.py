import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestRmsNormOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if 'normParam' in self.op_param.keys():
            normparam = self.op_param['normParam']
        else:
            normparam = self.op_param
        
        quant_type = normparam['quantType']
        layertype = self.op_param['layerType']

        eps = normparam['epsilon'] if 'epsilon' in normparam.keys() else 0.00001
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
            raise RuntimeError(f"RmsNorm: The divisor cannot be zero! Exception: {}".format(e))

        def rms_norm_quant(golden_output, beta):
            golden_output = golden_output.float()
            beta = beta.float()
            quant_scale = normparam['quantInputScale'] if 'quantInputScale' in normparam.keys() else 1
            quant_offset = normparam['quantInputOffset'] if 'quantInputOffset' in normparam.keys() else 0
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

