import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestConcatOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        concat_dim = self.op_param["concatDim"]
        axis_num = concat_dim if concat_dim >= 0 else concat_dim + len(in_tensors[0].size())
        golden_result = torch.cat(in_tensors, axis=axis_num)
        return [golden_result]

    def test(self):
        self.execute()
