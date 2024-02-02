import sys
import os
import unittest
import numpy as np
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


class TestRopeGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # x,128*32-->reshape x,32,128
        cos_list = [in_tensors[2][:x, :] for x in self.op_param['qSeqLen']]
        sin_list = [in_tensors[3][:x, :] for x in self.op_param['qSeqLen']]
        cos = torch.cat(cos_list, dim=0)
        sin = torch.cat(sin_list, dim=0)
        sin1 = sin[:, :64]
        sin2 = sin[:, 64:]
        rohqgsin = torch.cat((sin2, -sin1), dim=-1)
        q_grad = torch.zeros_like(in_tensors[0])
        bs = int(in_tensors[0].shape[1] / 128)
        for i in range(bs):
            q_grad[:, i * 128:(i + 1) * 128] = in_tensors[0][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)
    
        k_grad = torch.zeros_like(in_tensors[1])
        for i in range(bs):
            k_grad[:, i * 128:(i + 1) * 128] = in_tensors[1][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)
        return [q_grad, k_grad]

    def test(self):
        self.execute()