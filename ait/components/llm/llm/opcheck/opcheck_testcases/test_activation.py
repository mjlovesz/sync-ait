import os
import unittest
import torch
import torch_npu
import numpy as np

from llm.opcheck import operation_test


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors, _):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        try:
            float_result = 0.5 * float_in_tensors * (1 + torch.nn.functional.tanh(np.sqrt(2 / np.pi) * 
                                                    (float_in_tensors + 0.044715 * torch.pow(float_in_tensors, 3))))
        except ZeroDivisionError as e:
            raise RuntimeError(f"Gelu golden: The divisor cannot be zero! Exception: {}".format(e)) 
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result

    @staticmethod
    def fast_gelu_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        try:
            float_result = float_in_tensors * torch.exp(0.851 * (float_in_tensors - torch.abs(float_in_tensors))) / (1 + 
                            torch.exp(-1.702 * torch.abs(float_in_tensors)))
        except ZeroDivisionError as e:
            raise RuntimeError(f"Fast gelu golden: The divisor cannot be zero! Exception: {}".format(e))
        return float_result.half()

    @staticmethod
    def swish_golden(in_tensors, scale):
        float_in_tensors = in_tensors.float()
        try:
            float_result = float_in_tensors / (1 + torch.exp(-float_in_tensors * scale))
        except ZeroDivisionError as e:
            raise RuntimeError(f"Swish golden: The divisor cannot be zero! Exception: {}".format(e))
        return float_result.half()

    @staticmethod
    def log_golden(in_tensors, _):
        float_in_tensors = in_tensors.float()
        float_result = torch.log(float_in_tensors)
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result


class TestActivationOperation(operation_test.OperationTest):
    golden_func = {
        1: ActivationGolden.relu_golden,
        2: ActivationGolden.gelu_golden,
        3: ActivationGolden.fast_gelu_golden,
        4: ActivationGolden.swish_golden,
        5: ActivationGolden.log_golden
    } 

    def golden_calc(self, in_tensors):
        activation_type = self.op_param["activationType"]
        scale = self.op_param["scale"]
        golden_result = TestActivationOperation.golden_func[activation_type](in_tensors[0], scale)
        return [golden_result]

    def test(self):
        self.execute()