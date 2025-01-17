# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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

from testcase.refactor.test_node_common import create_node


class TestInitializer(unittest.TestCase):

    def test_initializer_get_value(self):
        ini = create_node('OnnxInitializer')
        self.assertTrue(np.array_equal(ini.value, np.array([[1, 2, 3, 4, 5]], dtype='int32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'int32')

    def test_initializer_set_value(self):
        ini = create_node('OnnxInitializer')
        ini.value = np.array([[7, 8, 9], [10, 11, 12]], dtype='float32')
        self.assertTrue(np.array_equal(ini.value, np.array([[7, 8, 9], [10, 11, 12]], dtype='float32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'float32')


if __name__ == "__main__":
    unittest.main()
