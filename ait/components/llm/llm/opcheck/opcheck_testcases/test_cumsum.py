import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestCumsumOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.cumsum(in_tensors[0], dim=self.op_param['axes'][0])
        return [golden_result]

    def test(self):
        self.execute()