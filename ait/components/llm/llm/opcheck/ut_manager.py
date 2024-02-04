# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import unittest
import queue
import torch
import torch_npu

from llm.opcheck.opcheck_testcases.test_activation import TestActivationOperation
from llm.opcheck.opcheck_testcases.test_all_gather import AllGatherOperationTest
from llm.opcheck.opcheck_testcases.test_all_reduce import AllReduceOperationTest
from llm.opcheck.opcheck_testcases.test_broadcast import BroadcastOperationTest
from llm.opcheck.opcheck_testcases.test_concat import TestConcatOperation
from llm.opcheck.opcheck_testcases.test_cumsum import TestCumsumOperation
from llm.opcheck.opcheck_testcases.test_elewise import TestElewiseAddOperation
from llm.opcheck.opcheck_testcases.test_fastsoftmax import TestFastSoftMaxOperation
from llm.opcheck.opcheck_testcases.test_fastsoftmaxgrad import TestFastSoftMaxGradOperation
from llm.opcheck.opcheck_testcases.test_fill import TestFillOperation
from llm.opcheck.opcheck_testcases.test_gather import TestGatherOperation
from llm.opcheck.opcheck_testcases.test_genattentionmask import TestElewiseSubOperation
from llm.opcheck.opcheck_testcases.test_kv_cache import TestKvCacheOperation
from llm.opcheck.opcheck_testcases.test_linear_activation import TestLinearActivationOperation
from llm.opcheck.opcheck_testcases.test_linear import TestLinearOperation
from llm.opcheck.opcheck_testcases.test_linear_activation_quant import TestLinearActivationQuantOperation
from llm.opcheck.opcheck_testcases.test_linear_quant import TestLinearQuantOperation
from llm.opcheck.opcheck_testcases.test_linear_sparse import TestLinearSparseOperation
from llm.opcheck.opcheck_testcases.test_matmul import TestMatmulOperation
from llm.opcheck.opcheck_testcases.test_pad import TestPadOperation
from llm.opcheck.opcheck_testcases.test_paged_attention import TestPagedAttentionAttentionOperation
from llm.opcheck.opcheck_testcases.test_repeat import TestRepeatOperation
from llm.opcheck.opcheck_testcases.test_reshape_and_cache import TestReshapeAndCacheOperation
from llm.opcheck.opcheck_testcases.test_rms_norm import TestRmsNormOperation
from llm.opcheck.opcheck_testcases.test_rope_grad import TestRopeGradOperation
from llm.opcheck.opcheck_testcases.test_rope import TestUnpadRopeOperation
from llm.opcheck.opcheck_testcases.test_self_attention import TestUnpadSelfAttentionOperation
from llm.opcheck.opcheck_testcases.test_set_value import TestSetValueOperation
from llm.opcheck.opcheck_testcases.test_slice import TestSliceOperation
from llm.opcheck.opcheck_testcases.test_softmax import TestSoftmaxOperation
from llm.opcheck.opcheck_testcases.test_sort import TestSortOperation
from llm.opcheck.opcheck_testcases.test_split import TestAddOperation
from llm.opcheck.opcheck_testcases.test_stridebatchmatmul import TestStridedBatchMatmulOperation
from llm.opcheck.opcheck_testcases.test_topk_topp_sampling import TestToppOperation
from llm.opcheck.opcheck_testcases.test_transpose import TestTransposeOperation
from llm.opcheck.opcheck_testcases.test_unpad import TestUnpadOperation
from llm.opcheck.opcheck_testcases.test_as_strided import TestAsStridedOperation
from llm.opcheck.opcheck_testcases.test_layer_norm import TestLayerNormOperation
from llm.opcheck.opcheck_testcases.test_linear_parallel import TestLinearParallelOperation
from llm.opcheck.opcheck_testcases.test_multinomial import TestMultinomialOperation
from llm.opcheck.opcheck_testcases.test_reduce import TestReduceOperation
from llm.opcheck.opcheck_testcases.test_transdata import TestTransdataOperation
from llm.opcheck.opcheck_testcases.test_where import TestWhereOperation


OP_NAME_DICT = {
    "ActivationOperation":TestActivationOperation,
    "AllGatherOperation":AllGatherOperationTest,
    "AllReduceOperation":AllReduceOperationTest,
    "BroadcastOperation":BroadcastOperationTest,
    "ConcatOperation":TestConcatOperation,
    "CumsumOperation":TestCumsumOperation,
    "ElewiseOperation":TestElewiseAddOperation,
    "FastSoftMaxOperation":TestFastSoftMaxOperation,
    "FastSoftMaxGradOperation":TestFastSoftMaxGradOperation,
    "FillOperation":TestFillOperation,
    "GatherOperation":TestGatherOperation,
    "GenAttentionMaskOperation":TestElewiseSubOperation,
    "KvCacheOperation":TestKvCacheOperation,
    "LinearOperation":TestLinearOperation,
    "LinearActivationOperation":TestLinearActivationOperation,
    "LinearActivationQuantOperation":TestLinearActivationQuantOperation,
    "LinearQuantOperation":TestLinearQuantOperation,
    "LinearSparseOperation":TestLinearSparseOperation,
    "MatmulOperation":TestMatmulOperation,
    "PadOperation":TestPadOperation,
    "PagedAttentionOperation":TestPagedAttentionAttentionOperation,
    "RepeatOperation":TestRepeatOperation,
    "ReshapeAndCacheOperation":TestReshapeAndCacheOperation,
    "RmsNormOperation":TestRmsNormOperation,
    "RopeOperation":TestUnpadRopeOperation,
    "RopeGradOperation":TestRopeGradOperation,
    "SelfAttentionOperation":TestUnpadSelfAttentionOperation,
    "SetValueOperation":TestSetValueOperation,
    "SliceOperation":TestSliceOperation,
    "SoftmaxOperation":TestSoftmaxOperation,
    "SortOperation":TestSortOperation,
    "SplitOperation":TestAddOperation,
    "StridedBatchMatmulOperation":TestStridedBatchMatmulOperation,
    "TopkToppSamplingOperation":TestToppOperation,
    "TransposeOperation":TestTransposeOperation,
    "UnpadOperation":TestUnpadOperation,
    "AsStridedOperation":TestAsStridedOperation,
    "LayerNormOperation":TestLayerNormOperation,
    "LinearParallelOperation":TestLinearParallelOperation,
    "MultinomialOperation":TestMultinomialOperation,
    "ReduceOperation":TestReduceOperation,
    "TransdataOperation":TestTransdataOperation,
    "WhereOperation":TestWhereOperation,
}


class UtManager:
    def __init__(self, excuted_ids=None):
        self.suite = unittest.TestSuite()
        self.cases = []
        self.excuted_ids = excuted_ids
    
    def add_case(self, case_info):
        op_name = case_info['op_name']
        if op_name not in OP_NAME_DICT.keys():
            #没有该op_name
            return False 
        else:
            if OP_NAME_DICT[op_name]:
                self.cases.append(case_info)
                return True
            else:
                #该算子用例未添加
                return False
      
    def add_cases_to_suite(self):
        for case_info in self.cases:
            op = OP_NAME_DICT[case_info['op_name']]
            self.suite.addTest(op.parametrize(optest_class=op, case_info=case_info, excuted_ids=self.excuted_ids))

    def excute_cases(self):
        self.add_cases_to_suite()

        #拉起测试套
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(self.suite)
