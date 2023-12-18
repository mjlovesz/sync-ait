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

import unittest

import numpy as np
import pytest

from llm.compare.acc_cmp import acc_compare, read_data, compare_data, compare_file


class TestAccCompare(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.test_data_path = "./test_data.npy"
        self.golden_data_path = "./golden_data.npy"
        self.dat_path = "./test_data.dat"

    def setUp(self) -> None:
        test_data = np.ones((2, 3)).astype(np.float32)
        golden_data = np.ones((2, 3)).astype(np.float32)
        np.save(self.test_data_path, test_data)
        np.save(self.golden_data_path, golden_data)
        np.save(self.dat_path, test_data)

    def test_acc_compare_when_data_file(self):
        acc_compare(self.golden_data_path, self.test_data_path)

    def test_read_data_when_npy(self):
        data = read_data(self.golden_data_path)
        golden = np.load(self.golden_data_path)
        assert (data == golden).all()

    def test_read_data_when_invalid_type(self):
        with pytest.raises(TypeError):
            read_data(self.dat_path)

    def test_compare_data_when_no_error(self):
        test_data = read_data(self.test_data_path)
        golden_data = read_data(self.golden_data_path)
        res = compare_data(test_data, golden_data)
        assert res == {'cosine_similarity': 1.0, 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                       'relative_euclidean_distance': 0.0}

    def test_compare_file(self):
        res = compare_file(self.golden_data_path, self.test_data_path)
        assert res == {'cosine_similarity': 1.0, 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                       'relative_euclidean_distance': 0.0}
