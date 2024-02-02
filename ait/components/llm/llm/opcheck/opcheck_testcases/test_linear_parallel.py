import os
import sys
import unittest
import torch
import torch_npu
import numpy as np
import torch.distributed as dist

from llm.opcheck import operation_test


class TestLinearParallelOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.matmul(in_tensor_0.to(torch.float32), in_tensor_1.to(torch.float32)).to(torch.float16)
        dist.all_reduce(golden_result, op=ReduceOp.SUM)
        torch.npu.synchronize()
        if in_tensors[2] is not None:
            golden_result = golden_result + in_tensors[2]

        return [golden_result]

    def test(self):
        self.execute()