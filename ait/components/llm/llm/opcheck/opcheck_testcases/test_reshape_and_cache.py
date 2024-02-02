import json
import math
import os
import random
import sys
import unittest

import numpy as np
import torch
import torch_npu

from llm.opcheck import operation_test


class TestReshapeAndCacheOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden = []
        for index in self.case_info['inplace_idx']:
            golden.append(in_tensors[index])
        return golden

    def test(self):
        self.execute_inplace()