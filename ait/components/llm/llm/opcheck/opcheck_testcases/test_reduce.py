import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestReduceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        op_type = self.op_param['reduceType']
        axis = self.op_param['axis']
        return [in_tensors[0].amax(axis)[0]] if op_type == 1 else [in_tensors[0].amin(axis)[0]]

    def test(self):
        self.execute()