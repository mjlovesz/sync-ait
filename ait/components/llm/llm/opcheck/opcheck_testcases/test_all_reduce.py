import os
import json
import unittest
import sys
import socket
import random
import torch
import torch_npu

from llm.opcheck import operation_test


class AllReduceOperationTest(operation_test.OperationTest):
    def lccl_sum_cal(self, in_tensors):
        cal_tensors = [0] * 8 
        for idx, in_tensor in enumerate(in_tensors):
            cal_tensors[idx] = in_tensor
        result = ((cal_tensors[0] + cal_tensors[1]) + (cal_tensors[2] + cal_tensors[3])) + ((cal_tensors[4] + 
                cal_tensors[5]) + (cal_tensors[6] + cal_tensors[7]))
        return [result]

    def sum_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)):
            result += in_tensors[i]
        return [result]

    def max_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)): 
            result = torch.max(result, in_tensors[i])
        return [result]

    def min_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)): 
            result = torch.min(result, in_tensors[i])
        return [result]

    def prod_cal(self, in_tensors):
        result = in_tensors[0]
        for i in range(1, len(in_tensors)):
            result = torch.mul(result, in_tensors[i])
        return [result]

    def golden_calc(self, in_tensors):
        all_reduce_type = self.op_param['allReduceType']
        backend = self.op_param['backend']
        logging.debug("backend: %s, allreduceType: %s", backend, all_reduce_type)
        logging.debug("env: %s", os.getenv("LCCL_DETERMINISTIC"))
        logging.debug("env: %s", os.getenv("HCCL_DETERMINISTIC"))
        
        if all_reduce_type == "sum":
            if backend == "lccl":
                golden = self.lccl_sum_cal(in_tensors)
            else:
                golden = self.sum_cal(in_tensors)
        elif all_reduce_type == "max":
            golden = self.max_cal(in_tensors)
        elif all_reduce_type == "min":
            golden = self.min_cal(in_tensors)
        elif all_reduce_type == "prod":
            golden = self.prod_cal(in_tensors)

        return golden

    def test_all_reduce(self):
        self.execute()