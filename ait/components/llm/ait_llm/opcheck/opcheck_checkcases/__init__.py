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

from ait_llm.opcheck.opcheck_checkcases.activation import OpcheckActivationOperation
from ait_llm.opcheck.opcheck_checkcases.all_gather import OpcheckAllGatherOperation
from ait_llm.opcheck.opcheck_checkcases.all_reduce import OpcheckAllReduceOperation
from ait_llm.opcheck.opcheck_checkcases.broadcast import OpcheckBroadcastOperation
from ait_llm.opcheck.opcheck_checkcases.concat import OpcheckConcatOperation
from ait_llm.opcheck.opcheck_checkcases.cumsum import OpcheckCumsumOperation
from ait_llm.opcheck.opcheck_checkcases.elewise import OpcheckElewiseAddOperation
from ait_llm.opcheck.opcheck_checkcases.fastsoftmax import OpcheckFastSoftMaxOperation
from ait_llm.opcheck.opcheck_checkcases.fastsoftmaxgrad import OpcheckFastSoftMaxGradOperation
from ait_llm.opcheck.opcheck_checkcases.fill import OpcheckFillOperation
from ait_llm.opcheck.opcheck_checkcases.gather import OpcheckGatherOperation
from ait_llm.opcheck.opcheck_checkcases.genattentionmask import OpcheckElewiseSubOperation
from ait_llm.opcheck.opcheck_checkcases.kv_cache import OpcheckKvCacheOperation
from ait_llm.opcheck.opcheck_checkcases.linear_activation import OpcheckLinearActivationOperation
from ait_llm.opcheck.opcheck_checkcases.linear import OpcheckLinearOperation
from ait_llm.opcheck.opcheck_checkcases.linear_activation_quant import OpcheckLinearActivationQuantOperation
from ait_llm.opcheck.opcheck_checkcases.linear_quant import OpcheckLinearQuantOperation
from ait_llm.opcheck.opcheck_checkcases.linear_sparse import OpcheckLinearSparseOperation
from ait_llm.opcheck.opcheck_checkcases.matmul import OpcheckMatmulOperation
from ait_llm.opcheck.opcheck_checkcases.pad import OpcheckPadOperation
from ait_llm.opcheck.opcheck_checkcases.paged_attention import OpcheckPagedAttentionAttentionOperation
from ait_llm.opcheck.opcheck_checkcases.repeat import OpcheckRepeatOperation
from ait_llm.opcheck.opcheck_checkcases.reshape_and_cache import OpcheckReshapeAndCacheOperation
from ait_llm.opcheck.opcheck_checkcases.rms_norm import OpcheckRmsNormOperation
from ait_llm.opcheck.opcheck_checkcases.rope_grad import OpcheckRopeGradOperation
from ait_llm.opcheck.opcheck_checkcases.rope import OpcheckUnpadRopeOperation
from ait_llm.opcheck.opcheck_checkcases.self_attention import OpcheckUnpadSelfAttentionOperation
from ait_llm.opcheck.opcheck_checkcases.set_value import OpcheckSetValueOperation
from ait_llm.opcheck.opcheck_checkcases.slice import OpcheckSliceOperation
from ait_llm.opcheck.opcheck_checkcases.softmax import OpcheckSoftmaxOperation
from ait_llm.opcheck.opcheck_checkcases.sort import OpcheckSortOperation
from ait_llm.opcheck.opcheck_checkcases.split import OpcheckAddOperation
from ait_llm.opcheck.opcheck_checkcases.stridebatchmatmul import OpcheckStridedBatchMatmulOperation
from ait_llm.opcheck.opcheck_checkcases.topk_topp_sampling import OpcheckToppOperation
from ait_llm.opcheck.opcheck_checkcases.transpose import OpcheckTransposeOperation
from ait_llm.opcheck.opcheck_checkcases.unpad import OpcheckUnpadOperation
from ait_llm.opcheck.opcheck_checkcases.as_strided import OpcheckAsStridedOperation
from ait_llm.opcheck.opcheck_checkcases.layer_norm import OpcheckLayerNormOperation
from ait_llm.opcheck.opcheck_checkcases.linear_parallel import OpcheckLinearParallelOperation
from ait_llm.opcheck.opcheck_checkcases.multinomial import OpcheckMultinomialOperation
from ait_llm.opcheck.opcheck_checkcases.reduce import OpcheckReduceOperation
from ait_llm.opcheck.opcheck_checkcases.transdata import OpcheckTransdataOperation
from ait_llm.opcheck.opcheck_checkcases.where import OpcheckWhereOperation


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