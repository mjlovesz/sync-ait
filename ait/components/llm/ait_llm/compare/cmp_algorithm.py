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
    if torch.all(golden_data == 0) and torch.all(my_data == 0):
        return 1.0, ''  # both are all 0, return similarity 1

    result = torch.cosine_similarity(golden_data.double(), my_data.double(), dim=0).item()  # Torch handle zero data
    return round(result, 10), ''  # Trunc to keeping only 10 decimals


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
    result = F.kl_div(F.log_softmax(my_data, dim=-1), F.softmax(golden_data, dim=-1), reduction="sum").item()
    return max(result, 0), ""


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
