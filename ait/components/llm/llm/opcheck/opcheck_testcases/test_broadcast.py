import os
import json
import unittest
import sys
import socket
import random
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

from llm.opcheck import operation_test


class BroadcastOperationTest(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        rank_root = self.op_param['rankRoot']
        golden_result = intensors[rank_root]
        return [golden_result]

    def test_broadcast(self):
        self.execute()