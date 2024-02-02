import json
import math
import os
import random
import sys
import unittest
import collections
import numpy as np
import torch
import torch_npu

from llm.opcheck import operation_test


class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if 'isSupportAlibi' in self.op_param:
            is_support_alibi = self.op_param["isSupportAlibi"]
        else:
            is_support_alibi = False
        
        if is_support_alibi:
            return [in_tensors[6]]
        else:
            return [in_tensors[5]]

    def test(self):
        self.execute()