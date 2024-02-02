import sys
import os
import unittest
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


class TestSoftmaxOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        softmax_func = torch.nn.Softmax(dim=self.op_param['axes'][0])
        return [softmax_func(in_tensors[0])]

    def test(self):
        self.execute()