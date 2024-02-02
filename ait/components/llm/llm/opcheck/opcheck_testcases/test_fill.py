import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestFillOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if self.op_param["withMask"]:
            golden_result = in_tensors[0].masked_fill_(in_tensors[1], self.op_param["value"][0])
        else:
            golden_result = torch.full(self.op_param["outDim"], self.op_param["value"][0], dtype=torch.float16)
        return [golden_result]

    def test(self):
        self.execute()