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

from llm.opcheck.opcheck_checkcases.activation import OpcheckActivationOperation
from llm.opcheck.opcheck_checkcases.all_gather import OpcheckAllGatherOperation
from llm.opcheck.opcheck_checkcases.all_reduce import OpcheckAllReduceOperation
from llm.opcheck.opcheck_checkcases.broadcast import OpcheckBroadcastOperation
from llm.opcheck.opcheck_checkcases.concat import OpcheckConcatOperation
from llm.opcheck.opcheck_checkcases.cumsum import OpcheckCumsumOperation
from llm.opcheck.opcheck_checkcases.elewise import OpcheckElewiseAddOperation
from llm.opcheck.opcheck_checkcases.fastsoftmax import OpcheckFastSoftMaxOperation
from llm.opcheck.opcheck_checkcases.fastsoftmaxgrad import OpcheckFastSoftMaxGradOperation
from llm.opcheck.opcheck_checkcases.fill import OpcheckFillOperation
from llm.opcheck.opcheck_checkcases.gather import OpcheckGatherOperation
from llm.opcheck.opcheck_checkcases.genattentionmask import OpcheckElewiseSubOperation
from llm.opcheck.opcheck_checkcases.kv_cache import OpcheckKvCacheOperation
from llm.opcheck.opcheck_checkcases.linear_activation import OpcheckLinearActivationOperation
from llm.opcheck.opcheck_checkcases.linear import OpcheckLinearOperation
from llm.opcheck.opcheck_checkcases.linear_activation_quant import OpcheckLinearActivationQuantOperation
from llm.opcheck.opcheck_checkcases.linear_quant import OpcheckLinearQuantOperation
from llm.opcheck.opcheck_checkcases.linear_sparse import OpcheckLinearSparseOperation
from llm.opcheck.opcheck_checkcases.matmul import OpcheckMatmulOperation
from llm.opcheck.opcheck_checkcases.pad import OpcheckPadOperation
from llm.opcheck.opcheck_checkcases.paged_attention import OpcheckPagedAttentionAttentionOperation
from llm.opcheck.opcheck_checkcases.repeat import OpcheckRepeatOperation
from llm.opcheck.opcheck_checkcases.reshape_and_cache import OpcheckReshapeAndCacheOperation
from llm.opcheck.opcheck_checkcases.rms_norm import OpcheckRmsNormOperation
from llm.opcheck.opcheck_checkcases.rope_grad import OpcheckRopeGradOperation
from llm.opcheck.opcheck_checkcases.rope import OpcheckUnpadRopeOperation
from llm.opcheck.opcheck_checkcases.self_attention import OpcheckUnpadSelfAttentionOperation
from llm.opcheck.opcheck_checkcases.set_value import OpcheckSetValueOperation
from llm.opcheck.opcheck_checkcases.slice import OpcheckSliceOperation
from llm.opcheck.opcheck_checkcases.softmax import OpcheckSoftmaxOperation
from llm.opcheck.opcheck_checkcases.sort import OpcheckSortOperation
from llm.opcheck.opcheck_checkcases.split import OpcheckAddOperation
from llm.opcheck.opcheck_checkcases.stridebatchmatmul import OpcheckStridedBatchMatmulOperation
from llm.opcheck.opcheck_checkcases.topk_topp_sampling import OpcheckToppOperation
from llm.opcheck.opcheck_checkcases.transpose import OpcheckTransposeOperation
from llm.opcheck.opcheck_checkcases.unpad import OpcheckUnpadOperation
from llm.opcheck.opcheck_checkcases.as_strided import OpcheckAsStridedOperation
from llm.opcheck.opcheck_checkcases.layer_norm import OpcheckLayerNormOperation
from llm.opcheck.opcheck_checkcases.linear_parallel import OpcheckLinearParallelOperation
from llm.opcheck.opcheck_checkcases.multinomial import OpcheckMultinomialOperation
from llm.opcheck.opcheck_checkcases.reduce import OpcheckReduceOperation
from llm.opcheck.opcheck_checkcases.transdata import OpcheckTransdataOperation
from llm.opcheck.opcheck_checkcases.where import OpcheckWhereOperation


OP_NAME_DICT = dict({
    "ActivationOperation":OpcheckActivationOperation,
    "AllGatherOperation":OpcheckAllGatherOperation,
    "AllReduceOperation":OpcheckAllReduceOperation,
    "BroadcastOperation":OpcheckBroadcastOperation,
    "ConcatOperation":OpcheckConcatOperation,
    "CumsumOperation":OpcheckCumsumOperation,
    "ElewiseOperation":OpcheckElewiseAddOperation,
    "FastSoftMaxOperation":OpcheckFastSoftMaxOperation,
    "FastSoftMaxGradOperation":OpcheckFastSoftMaxGradOperation,
    "FillOperation":OpcheckFillOperation,
    "GatherOperation":OpcheckGatherOperation,
    "GenAttentionMaskOperation":OpcheckElewiseSubOperation,
    "KvCacheOperation":OpcheckKvCacheOperation,
    "LinearOperation":OpcheckLinearOperation,
    "LinearActivationOperation":OpcheckLinearActivationOperation,
    "LinearActivationQuantOperation":OpcheckLinearActivationQuantOperation,
    "LinearQuantOperation":OpcheckLinearQuantOperation,
    "LinearSparseOperation":OpcheckLinearSparseOperation,
    "MatmulOperation":OpcheckMatmulOperation,
    "PadOperation":OpcheckPadOperation,
    "PagedAttentionOperation":OpcheckPagedAttentionAttentionOperation,
    "RepeatOperation":OpcheckRepeatOperation,
    "ReshapeAndCacheOperation":OpcheckReshapeAndCacheOperation,
    "RmsNormOperation":OpcheckRmsNormOperation,
    "RopeOperation":OpcheckUnpadRopeOperation,
    "RopeGradOperation":OpcheckRopeGradOperation,
    "SelfAttentionOperation":OpcheckUnpadSelfAttentionOperation,
    "SetValueOperation":OpcheckSetValueOperation,
    "SliceOperation":OpcheckSliceOperation,
    "SoftmaxOperation":OpcheckSoftmaxOperation,
    "SortOperation":OpcheckSortOperation,
    "SplitOperation":OpcheckAddOperation,
    "StridedBatchMatmulOperation":OpcheckStridedBatchMatmulOperation,
    "TopkToppSamplingOperation":OpcheckToppOperation,
    "TransposeOperation":OpcheckTransposeOperation,
    "UnpadOperation":OpcheckUnpadOperation,
    "AsStridedOperation":OpcheckAsStridedOperation,
    "LayerNormOperation":OpcheckLayerNormOperation,
    "LinearParallelOperation":OpcheckLinearParallelOperation,
    "MultinomialOperation":OpcheckMultinomialOperation,
    "ReduceOperation":OpcheckReduceOperation,
    "TransdataOperation":OpcheckTransdataOperation,
    "WhereOperation":OpcheckWhereOperation,
})