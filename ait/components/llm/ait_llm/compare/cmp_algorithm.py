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
import torch
from torch.nn import functional as F

from ait_llm.common.log import logger


FLOAT_EPSILON = torch.finfo(torch.float).eps
NAN = 'NaN'


def cosine_similarity(golden_data: torch.Tensor, my_data: torch.Tensor):
    my_data_norm = torch.norm(my_data, dim=-1, keepdim=True, p=2)
    golden_data_norm = torch.norm(golden_data, dim=-1, keepdim=True, p=2)
    if my_data_norm <= FLOAT_EPSILON and golden_data_norm < FLOAT_EPSILON:
        return "1.0", ''
    elif my_data_norm ** 0.5 <= FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the values in my_data are zeros.'
        logger.warning(message)
        return NAN, message
    elif golden_data_norm ** 0.5 <= FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the values in golden_data are zeros.'
        logger.warning(message)
        return NAN, message

    result = torch.cosine_similarity(golden_data, my_data, dim=0)
    return '{:.6f}'.format(result), ''


def max_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=golden_data.dtype),
    ).max()
    return result.item(), ''


def mean_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).mean()
    return result.item(), ''


def max_absolute_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data - golden_data),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).max()
    return result.item(), ''


def mean_absolute_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data - golden_data),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        torch.tensor(0, dtype=my_data.dtype),
    ).mean()
    return result.item(), ''


def kl_divergence(golden_data: torch.Tensor, my_data: torch.Tensor):
    try:
        norm_xx = (my_data - my_data.min()) / (my_data.max() - my_data.min()) + FLOAT_EPSILON
        norm_yy = (golden_data - golden_data.min()) / (golden_data.max() - golden_data.min()) + FLOAT_EPSILON
    except ZeroDivisionError as e:
        return "", "The max value and min value of my_data or golden_data is equal."

    norm_xx /= norm_xx.sum()
    norm_yy /= norm_yy.sum()
    return (norm_xx * (norm_xx / norm_yy).log()).sum().item(), ""


def relative_euclidean_distance(golden_data: torch.Tensor, my_data: torch.Tensor):
    ground_truth_square_num = (golden_data ** 2).sum()
    if ground_truth_square_num ** 0.5 <= FLOAT_EPSILON:
        return 0.0, ''

    result = ((my_data - golden_data) ** 2).sum() / ground_truth_square_num
    return result.item(), ''


CMP_ALG_MAP = {
    "cosine_similarity": cosine_similarity,
    "max_relative_error": max_relative_error,
    "mean_relative_error": mean_relative_error,
    "max_absolute_error": max_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "kl_divergence": kl_divergence,
    "relative_euclidean_distance": relative_euclidean_distance,
}
