import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestAsStridedOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.as_strided(in_tensors[0], self.op_param['size'], 
                                        self.op_param['stride'], self.op_param['offset'][0])
        return [golden_result]

    def test(self):
        self.execute()