import os
import sys
import unittest
import torch
import torch_npu
import numpy as np

from llm.opcheck import operation_test


class TestLinearQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensors[0].to(torch.int32), in_tensors[1].to(torch.int32))
        if self.op_param["hasBias"]:
            golden_result = golden_result + in_tensors[2]
        golden_result = golden_result * in_tensors[3]
        golden_result = golden_result.to(torch.float16)
        return [golden_result.npu()]

    def test(self):
        self.execute()