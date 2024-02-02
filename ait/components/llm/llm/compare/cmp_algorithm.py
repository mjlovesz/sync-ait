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

from llm.common.log import logger


# FLOAT_EPSILON = np.finfo(float).eps
FLOAT_EPSILON = torch.finfo(float).eps
# np.seterr(divide='ignore', invalid='ignore')  # ignore `invalid value encountered in true_divide` warning
NAN = 'NaN'


def cosine_similarity(golden_data: torch.Tensor, my_data: torch.Tensor):
    # my_data_norm = np.linalg.norm(my_data, axis=-1, keepdims=True)
    my_data_norm = torch.norm(my_data, p=2)
    # golden_data_norm = np.linalg.norm(golden_data, axis=-1, keepdims=True)
    golden_data_norm = torch.norm(golden_data, p=2)
    if my_data_norm <= FLOAT_EPSILON and golden_data_norm < FLOAT_EPSILON:
        return "1.0", ''
    elif my_data_norm ** 0.5 <= FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the my_data is zero'
        logger.warning(message)
        return NAN, message
    elif golden_data_norm ** 0.5 <= FLOAT_EPSILON:
        message = 'Cannot compare by Cosine Similarity. All the golden_data is zero'
        logger.warning(message)
        return NAN, message

    result = (my_data / my_data_norm) @ (golden_data / golden_data_norm)
    return '{:.6f}'.format(result), ''


def max_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        0,
    ).max()
    return result, ''


def mean_relative_error(golden_data: torch.Tensor, my_data: torch.Tensor):
    result = torch.where(
        torch.abs(golden_data) > FLOAT_EPSILON,
        torch.abs(my_data / golden_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        0,
    ).mean()
    return result, ''


def relative_euclidean_distance(golden_data: torch.Tensor, my_data: torch.Tensor):
    ground_truth_square_num = (golden_data ** 2).sum()
    if ground_truth_square_num ** 0.5 <= FLOAT_EPSILON:
        result = 0.0
    else:
        result = ((my_data - golden_data) ** 2).sum() / ground_truth_square_num
    return result, ''


CMP_ALG_MAP = {
    "cosine_similarity": cosine_similarity,
    "max_relative_error": max_relative_error,
    "mean_relative_error": mean_relative_error,
    "relative_euclidean_distance": relative_euclidean_distance
}
