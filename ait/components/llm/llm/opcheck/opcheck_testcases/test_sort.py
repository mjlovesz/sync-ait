import sys
import os
import json
import unittest
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


class TestSortOperation(operation_test.OperationTest):    
    def golden_calc(self, in_tensors):
        values, indices = torch.topk(in_tensors[0], k=self.op_param["num"][0], largest=True)
        return [values, indices.int()]

    def test_3d_float(self):
        self.execute()