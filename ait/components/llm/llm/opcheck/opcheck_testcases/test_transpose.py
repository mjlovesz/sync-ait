import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestTransposeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        perm = self.op_param["perm"]
        golden_result = in_tensors[0].permute(perm)
        return [golden_result]

    def test_2d_float(self):
        self.execute()