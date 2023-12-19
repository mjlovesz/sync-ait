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
import os.path

import pytest
import numpy as np

from llm.compare.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, \
    relative_euclidean_distance


@pytest.fixture(scope='module', autouse=True)
def golden_data():
    golden_data = np.ones((2, 3)).astype(np.float32)
    yield golden_data


@pytest.fixture(scope='module', autouse=True)
def test_data():
    test_data = np.ones((2, 3)).astype(np.float32)
    yield test_data


def test_cosine_similarity(golden_data, test_data):
    res = cosine_similarity(golden_data.reshape(-1), test_data.reshape(-1))
    assert res == '1.000000'


def test_max_relative_error(golden_data, test_data):
    res = max_relative_error(golden_data, test_data)
    assert res == 0.0


def test_mean_relative_error(golden_data, test_data):
    res = mean_relative_error(golden_data, test_data)
    assert res == 0.0


def test_relative_euclidean_distance(golden_data, test_data):
    res = relative_euclidean_distance(golden_data, test_data)
    assert res == 0.0