import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestElewiseSubOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        out = []
        for i, s in enumerate(self.op_param['seqLen']):
            for _ in range(self.op_param["headNum"]):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def test_2d_half(self):
        self.execute()