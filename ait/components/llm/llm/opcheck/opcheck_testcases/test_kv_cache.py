import sys
import os
import time
import json
import unittest
import torch
import torch_npu
import numpy as np

from llm.opcheck import operation_test


class TestKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden = []
        for index in self.case_info['inplace_idx']:
            golden.append(in_tensors[index])
        return golden

    def test(self):
        self.execute_inplace()