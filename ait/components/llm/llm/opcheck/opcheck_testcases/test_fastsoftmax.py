import sys
import os
import unittest
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


class TestFastSoftMaxOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0]
        seq_len_list = self.op_param['qSeqLen']
        head_num_imm = self.op_param['headNum']
        golden = torch.empty_like(data_input)

        start = 0
        for seq_len in seq_len_list:
            end = start + head_num_imm * seq_len * seq_len
            cur_data_input = data_input[start:end].reshape(-1, seq_len)
            cur_golden = torch.softmax(cur_data_input.to(torch.float32), dim=-1).to(torch.float16)
            golden[start:end] = cur_golden.reshape(-1)
            start = end
        return [golden]

    def test_fastsoftmax(self):
        self.execute()