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
import numpy as np

from msquickcmp.common.utils import logger


FLOAT_EPSILON = np.finfo(float).eps
NAN = 'NaN'


def cosine_similarity(pta_data: np.ndarray, acl_data: np.ndarray):
    acl_data_norm = np.linalg.norm(acl_data, axis=-1, keepdims=True)
    pta_data_norm = np.linalg.norm(pta_data, axis=-1, keepdims=True)
    if acl_data_norm <= FLOAT_EPSILON and pta_data_norm < FLOAT_EPSILON:
        return "1.0", ""
    elif acl_data_norm ** 0.5 <= FLOAT_EPSILON:
        logger.warning('Cannot compare by Cosine Similarity. All the acl_data is zero')
        return NAN
    elif pta_data_norm ** 0.5 <= FLOAT_EPSILON:
        logger.warning('Cannot compare by Cosine Similarity. All the pta_data is zero')
        return NAN
    result = (acl_data / acl_data_norm) @ (pta_data / pta_data_norm)
    return '{:.6f}'.format(result)


def max_relative_error(pta_data: np.ndarray, acl_data: np.ndarray):
    result = np.where(
        np.abs(pta_data) > FLOAT_EPSILON,
        np.abs(acl_data / pta_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        0,
        ).max()
    return result


def mean_relative_error(pta_data: np.ndarray, acl_data: np.ndarray):
    result = np.where(
        np.abs(pta_data) > FLOAT_EPSILON,
        np.abs(acl_data / pta_data - 1),  # abs(aa - bb) / abs(bb) -> abs(aa / bb - 1)
        0,
    ).mean()
    return result


def relative_euclidean_distance(pta_data: np.ndarray, acl_data: np.ndarray):
    ground_truth_square_num = (pta_data ** 2).sum()
    if ground_truth_square_num ** 0.5 <= FLOAT_EPSILON:
        result = 0.0
    else:
        result = ((acl_data - pta_data) ** 2).sum() / ground_truth_square_num
    return result


cmp_alg_map = {
    "cosine_similarity": cosine_similarity,
    "max_relative_error": max_relative_error,
    "mean_relative_error": mean_relative_error,
    "relative_euclidean_distance": relative_euclidean_distance
}
