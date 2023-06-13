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
from unittest import mock

from aie_runtime.bean.config import ConvertConfig
from aie_runtime.core import Convert


class TestConvert(unittest.TestCase):

    def test_convert(self):
        config = ConvertConfig("test.onnx", "test.om", "Ascend310")
        convert = Convert(config)
        convert.execute_command = mock.Mock(return_value="Execute command success.")
        convert.convert_model()
