import sys
import os
import unittest
from ctypes import CDLL
import numpy as np
import torch
import torch_npu
import torch.nn as nn

from llm.opcheck import operation_test


class TestToppOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        rand_seed = self.op_param["randSeed"]
        topk = self.op_param["topk"]
        libc = CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]
 
        probs = in_tensors[0].cpu().numpy()
        topp = in_tensors[1].cpu().numpy()
        probs_sorted = np.sort(probs, axis=-1)[..., ::-1][..., :topk]
        try:
            probs_div_sorted = probs_sorted / topp
        except ZeroDivisionError as e:
            raise RuntimeError(f"Topp: The divisor cannot be zero! Exception: {}".format(e))    
        indices_sorted = np.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
        probs_sorted_sumed = np.cumsum(probs_sorted, axis=-1, dtype=np.float16).astype(np.float16)
        mask = np.zeros_like(probs_sorted_sumed, dtype=np.int32)
        mask[probs_sorted_sumed <= topp] = 1
        probs_div_sorted *= mask
        probs_div_sorted_sumed = np.cumsum(probs_div_sorted, axis=-1, dtype=np.float16).astype(np.float16)
        multinomial_probs = probs_div_sorted_sumed.astype(np.float32) < rand_list[0]
        first_true_indeces = np.argmax(~multinomial_probs, axis=-1)
        for i in range(probs.shape[0]):
            multinomial_probs[i, first_true_indeces[i]:] = False
        indices_sorted_sampled = np.sum(multinomial_probs, axis=-1, keepdims=True)
        indices_sorted_sampled[indices_sorted_sampled >= topk] = 0
        indices_sampled = np.take_along_axis(indices_sorted, indices_sorted_sampled, axis=-1)
        probs_sampled = np.take_along_axis(probs_sorted, indices_sorted_sampled, axis=-1)
        return [torch.from_numpy(indices_sampled.astype(np.int32)),
                torch.from_numpy(probs_sampled.astype(np.float16))]

    
    def test(self):
        self.execute()