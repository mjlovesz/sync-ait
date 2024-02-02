import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestWhereOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.where(in_tensors[0].bool(), in_tensors[1], in_tensors[2])
        return [golden_result]

    def test(self):
        self.execute()