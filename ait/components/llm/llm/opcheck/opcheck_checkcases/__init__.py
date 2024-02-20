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

from llm.opcheck.opcheck_checkcases.activation import TestActivationOperation
from llm.opcheck.opcheck_checkcases.all_gather import AllGatherOperationTest
from llm.opcheck.opcheck_checkcases.all_reduce import AllReduceOperationTest
from llm.opcheck.opcheck_checkcases.broadcast import BroadcastOperationTest
from llm.opcheck.opcheck_checkcases.concat import TestConcatOperation
from llm.opcheck.opcheck_checkcases.cumsum import TestCumsumOperation
from llm.opcheck.opcheck_checkcases.elewise import TestElewiseAddOperation
from llm.opcheck.opcheck_checkcases.fastsoftmax import TestFastSoftMaxOperation
from llm.opcheck.opcheck_checkcases.fastsoftmaxgrad import TestFastSoftMaxGradOperation
from llm.opcheck.opcheck_checkcases.fill import TestFillOperation
from llm.opcheck.opcheck_checkcases.gather import TestGatherOperation
from llm.opcheck.opcheck_checkcases.genattentionmask import TestElewiseSubOperation
from llm.opcheck.opcheck_checkcases.kv_cache import TestKvCacheOperation
from llm.opcheck.opcheck_checkcases.linear_activation import TestLinearActivationOperation
from llm.opcheck.opcheck_checkcases.linear import TestLinearOperation
from llm.opcheck.opcheck_checkcases.linear_activation_quant import TestLinearActivationQuantOperation
from llm.opcheck.opcheck_checkcases.linear_quant import TestLinearQuantOperation
from llm.opcheck.opcheck_checkcases.linear_sparse import TestLinearSparseOperation
from llm.opcheck.opcheck_checkcases.matmul import TestMatmulOperation
from llm.opcheck.opcheck_checkcases.pad import TestPadOperation
from llm.opcheck.opcheck_checkcases.paged_attention import TestPagedAttentionAttentionOperation
from llm.opcheck.opcheck_checkcases.repeat import TestRepeatOperation
from llm.opcheck.opcheck_checkcases.reshape_and_cache import TestReshapeAndCacheOperation
from llm.opcheck.opcheck_checkcases.rms_norm import TestRmsNormOperation
from llm.opcheck.opcheck_checkcases.rope_grad import TestRopeGradOperation
from llm.opcheck.opcheck_checkcases.rope import TestUnpadRopeOperation
from llm.opcheck.opcheck_checkcases.self_attention import TestUnpadSelfAttentionOperation
from llm.opcheck.opcheck_checkcases.set_value import TestSetValueOperation
from llm.opcheck.opcheck_checkcases.slice import TestSliceOperation
from llm.opcheck.opcheck_checkcases.softmax import TestSoftmaxOperation
from llm.opcheck.opcheck_checkcases.sort import TestSortOperation
from llm.opcheck.opcheck_checkcases.split import TestAddOperation
from llm.opcheck.opcheck_checkcases.stridebatchmatmul import TestStridedBatchMatmulOperation
from llm.opcheck.opcheck_checkcases.topk_topp_sampling import TestToppOperation
from llm.opcheck.opcheck_checkcases.transpose import TestTransposeOperation
from llm.opcheck.opcheck_checkcases.unpad import TestUnpadOperation
from llm.opcheck.opcheck_checkcases.as_strided import TestAsStridedOperation
from llm.opcheck.opcheck_checkcases.layer_norm import TestLayerNormOperation
from llm.opcheck.opcheck_checkcases.linear_parallel import TestLinearParallelOperation
from llm.opcheck.opcheck_checkcases.multinomial import TestMultinomialOperation
from llm.opcheck.opcheck_checkcases.reduce import TestReduceOperation
from llm.opcheck.opcheck_checkcases.transdata import TestTransdataOperation
from llm.opcheck.opcheck_checkcases.where import TestWhereOperation


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