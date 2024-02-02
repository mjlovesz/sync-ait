import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestRepeatOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        outtensor = in_tensors[0].repeat(self.op_param["multiples"])
        return [outtensor]

    def test(self):
        self.execute()
