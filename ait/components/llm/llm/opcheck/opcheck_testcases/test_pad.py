import sys
import os
import unittest
import torch
import torch_npu
import numpy as np

from llm.opcheck import operation_test


class TestPadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        tmp_out = in_tensors[0]
        padding_offset = in_tensors[1]
        seq_len = in_tensors[2]
        input_ids = in_tensors[3]
        batch = input_ids.shape[0]
        hidden_dim = tmp_out.shape[1]
        max_seq_len = input_ids.shape[1]
 
        golden_result = np.zeros((batch, hidden_dim)).astype(np.float16)
        temp_val = 0
        for i in range(batch):
            temp_val = temp_val + seq_len[i][0]
            golden_result[i] = tmp_out[temp_val - 1]
        golden_result = torch.from_numpy(golden_result)
        return [golden_result]
    
    def test(self):
        self.execute()