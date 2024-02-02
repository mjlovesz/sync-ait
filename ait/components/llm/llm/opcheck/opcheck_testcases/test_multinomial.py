import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestMultinomialOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        samples = self.op_param["numSamples"]
        rand_seed = self.op_param["randSeed"]
        input0 = in_tensors[0].cpu().numpy()
        libc = CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]
        ret = np.zeros(shape=(input0.shape[0], samples))

        sum_list = np.cumsum(input0, axis=-1, dtype=np.float16).astype(np.float16)
        iter_list = [(j, i) 
                    for j in range(input0.shape[0]) 
                    for i in range(input0.shape[1])]
        for z in range(samples):
            for j, i in iter_list:
                if (sum_list[j][i] > rand_list[z]):
                    ret[j][z] = i
                    break
        return [torch.from_numpy(ret.astype(np.int32)).contiguous()]

    def test(self):
        self.execute()