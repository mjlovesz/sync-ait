import sys
import os
import json
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestSetValueOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = [in_tensors[0].clone(), in_tensors[1].clone()]
        for i in range(len(self.op_param["starts"])):
            golden_result[0][self.op_param["starts"][i]:self.op_param["ends"][i]].copy_(in_tensors[1])
        return golden_result

    def test(self):
        self.execute()