import sys
import os
import unittest
import torch
import torch_npu

from llm.opcheck import operation_test


class TestSliceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        offset_list = self.op_param['offsets']
        size_list = self.op_param['size']
        for index, offset in enumerate(offset_list):
            offset_list[index] = offset if offset >= 0 else offset + in_tensors[0].shape[index]
        for index, size in enumerate(size_list):
            size_list[index] = size if size != -1 else in_tensors[0].shape[index] - offset_list[index]
        if len(offset_list) == 3:
            return [in_tensors[0][offset_list[0] : offset_list[0] + size_list[0], 
                    offset_list[1] : offset_list[1] + size_list[1], offset_list[2] : offset_list[2] + size_list[2]]]
        else:
            return [in_tensors[0][offset_list[0] : offset_list[0] + size_list[0], 
                    offset_list[1] : offset_list[1] + size_list[1], offset_list[2] : offset_list[2] + size_list[2], 
                    offset_list[3] : offset_list[3] + size_list[3]]]



    def test(self):
        self.execute()