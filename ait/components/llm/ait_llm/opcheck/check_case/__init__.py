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

from ait_llm.opcheck.check_case.activation import OpcheckActivationOperation
from ait_llm.opcheck.check_case.all_gather import OpcheckAllGatherOperation
from ait_llm.opcheck.check_case.all_reduce import OpcheckAllReduceOperation
from ait_llm.opcheck.check_case.broadcast import OpcheckBroadcastOperation
from ait_llm.opcheck.check_case.concat import OpcheckConcatOperation
from ait_llm.opcheck.check_case.cumsum import OpcheckCumsumOperation
from ait_llm.opcheck.check_case.elewise import OpcheckElewiseAddOperation
from ait_llm.opcheck.check_case.fastsoftmax import OpcheckFastSoftMaxOperation
from ait_llm.opcheck.check_case.fastsoftmaxgrad import OpcheckFastSoftMaxGradOperation
from ait_llm.opcheck.check_case.fill import OpcheckFillOperation
from ait_llm.opcheck.check_case.gather import OpcheckGatherOperation
from ait_llm.opcheck.check_case.genattentionmask import OpcheckElewiseSubOperation
from ait_llm.opcheck.check_case.kv_cache import OpcheckKvCacheOperation
from ait_llm.opcheck.check_case.linear import OpcheckLinearOperation
from ait_llm.opcheck.check_case.linear_sparse import OpcheckLinearSparseOperation
from ait_llm.opcheck.check_case.matmul import OpcheckMatmulOperation
from ait_llm.opcheck.check_case.pad import OpcheckPadOperation
from ait_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation
from ait_llm.opcheck.check_case.repeat import OpcheckRepeatOperation
from ait_llm.opcheck.check_case.reshape_and_cache import OpcheckReshapeAndCacheOperation
from ait_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation
from ait_llm.opcheck.check_case.rope_grad import OpcheckRopeGradOperation
from ait_llm.opcheck.check_case.rope import OpcheckUnpadRopeOperation
from ait_llm.opcheck.check_case.self_attention import OpcheckUnpadSelfAttentionOperation
from ait_llm.opcheck.check_case.set_value import OpcheckSetValueOperation
from ait_llm.opcheck.check_case.slice import OpcheckSliceOperation
from ait_llm.opcheck.check_case.softmax import OpcheckSoftmaxOperation
from ait_llm.opcheck.check_case.sort import OpcheckSortOperation
from ait_llm.opcheck.check_case.split import OpcheckAddOperation
from ait_llm.opcheck.check_case.stridebatchmatmul import OpcheckStridedBatchMatmulOperation
from ait_llm.opcheck.check_case.topk_topp_sampling import OpcheckToppOperation
from ait_llm.opcheck.check_case.transpose import OpcheckTransposeOperation
from ait_llm.opcheck.check_case.unpad import OpcheckUnpadOperation
from ait_llm.opcheck.check_case.as_strided import OpcheckAsStridedOperation
from ait_llm.opcheck.check_case.layer_norm import OpcheckLayerNormOperation
from ait_llm.opcheck.check_case.linear_parallel import OpcheckLinearParallelOperation
from ait_llm.opcheck.check_case.multinomial import OpcheckMultinomialOperation
from ait_llm.opcheck.check_case.reduce import OpcheckReduceOperation
from ait_llm.opcheck.check_case.transdata import OpcheckTransdataOperation
from ait_llm.opcheck.check_case.where import OpcheckWhereOperation


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