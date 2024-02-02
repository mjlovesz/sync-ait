import os
import json
import unittest
import sys
import torch
import torch_npu

from llm.opcheck import operation_test


class AllGatherOperationTest(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.stack(in_tensors, dim=0)
        return [golden_result]        

    def test_all_gather(self):
        self.excute()