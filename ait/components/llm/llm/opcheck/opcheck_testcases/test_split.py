import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        split_output = torch.chunk(in_tensors[0], chunks=self.op_param['splitNum'], dim=self.op_param['splitDim'])
        return split_output

    def test(self):
        self.execute()