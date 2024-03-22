# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
from ait_llm.common.log import logger

import torch
import torch.nn.functional as function
import torch.distributed as dist

HOOK_OPS = {}


def add_torch_ops():
    torch_hooks = {
        function: [
            "threshold",
            "threshold_",
            "relu",
            "relu_",
            "glu",
            "hardtanh",
            "hardtanh_",
            "relu6",
            "elu",
            "elu_",
            "selu",
            "selu_",
            "celu"
            "celu_",
            "leaky_relu",
            "leaky_relu_",
            "prelu",
            "rrelu",
            "rrelu_",
            "logsigmoid",
            "gelu",
            "hardshrink",
            "tanhshrink",
            "softsign",
            "softplus",
            "softmin",
            "softmax",
            "gumbel_softmax",
            "log_softmax"
            "softshrink",
            "tanh",
            "sigmoid",
            "hardsigmoid",
            "silu",
            "hardswish",
            "pixel_shuffle",
            "pixel_unshuffle",
            "channel_shuffle",
            "upsample_nearest",
            "upsample_bilinear",
            "grid_sample",
            "affine_grid",
            "pdist",
            "one_hot"
        ],
        torch: [
            "relu",
            "relu_",
            "rrelu",
            "rrelu_",
            "selu",
            "selu_",
            "sigmoid",
            "sigmoid_",
            "softmax",
            "tanh",
            "tanh_",
            "topk",
        ],
        dist: [
            "send",
            "recv",
            "broadcast",
            "all_reduce",
            "reduce",
            "all_gather",
            "gather",
            "isend",
            "irecv",
            "scatter",
            "reduce_scatter",
        ]
    }
    HOOK_OPS.update(torch_hooks)


def add_torch_npu_ops():
    try:
        import torch_npu
    except ImportError:
        logger.warning("torch_npu is not installed.")
        return
        
    torch_npu_hooks = [
        "fast_gelu",
        "npu_mish",
        "npu_scaled_masked_softmax",
        "npu_dropout_with_add_softmax",
        "npu_random_choice_with_mask",
        "npu_roi_align",
        "npu_roi_alignbk",
        "npu_all_gather_base_mm"
    ]
    HOOK_OPS[torch_npu] = torch_npu_hooks
