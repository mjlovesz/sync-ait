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
import os
import unittest

import numpy as np
import pytest

from llm.compare.acc_cmp import acc_compare, read_data, compare_data, compare_file


@pytest.fixture(scope='module')
def golden_data_file():
    golden_data = np.ones((2, 3)).astype(np.float32)
    golden_data_path = "./golden_data.npy"
    np.save(golden_data_path, golden_data)
    yield golden_data_path
    if os.path.exists(golden_data_path):
        os.remove(golden_data_path)


@pytest.fixture(scope='module')
def test_data_file():
    test_data = np.ones((2, 3)).astype(np.float32)
    test_data_path = "./test_data.npy"
    np.save(test_data_path, test_data)
    yield test_data_path
    if os.path.exists(test_data_path):
        os.remove(test_data_path)


@pytest.fixture(scope='module')
def test_dat_path():
    test_data = np.ones((2, 3)).astype(np.float32)
    test_data_path = "./test_data.dat"
    np.save(test_data_path, test_data)
    yield test_data_path
    if os.path.exists(test_data_path):
        os.remove(test_data_path)


def test_acc_compare_when_data_file(golden_data_file, test_data_file):
    acc_compare(golden_data_file, test_data_file)


def test_read_data_when_npy(golden_data_file, test_data_file):
    data = read_data(test_data_file)
    golden = np.load(golden_data_file)
    assert (data == golden).all()


def test_read_data_when_invalid_type(test_dat_path):
    with pytest.raises(TypeError):
        read_data(test_dat_path)


def test_compare_data_when_no_error(golden_data_file, test_data_file):
    test_data = read_data(test_data_file)
    golden_data = read_data(golden_data_file)
    res = compare_data(test_data, golden_data)
    assert res == {'cosine_similarity': '1.000000', 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                   'relative_euclidean_distance': 0.0}


def test_compare_file(golden_data_file, test_data_file):
    res = compare_file(golden_data_file, test_data_file)
    assert res == {'cosine_similarity': '1.000000', 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                   'relative_euclidean_distance': 0.0}
