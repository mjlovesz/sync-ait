import sys
import os
import unittest
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
    x.requires_grad = True
    y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
    y.retain_grad()
    w = torch.randn_like(x).to(torch.float32)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))


class TestFastSoftMaxGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        batch_size_imm = 4
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm, ]).to(torch.int64)
        seq_len = torch.randint(100, 300, [batch_size_imm, ]).to(torch.int32)
        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        golden = torch.cat(golden_list)

        return [golden]

    def test_fastsoftmaxgrad(self):
        self.execute()