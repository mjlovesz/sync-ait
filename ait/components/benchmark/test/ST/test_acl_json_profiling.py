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

import math
import os
import shutil
import json
import sys
import stat
import logging

import pytest
from test_common import TestCommonClass

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    @classmethod
    def get_resnet50_om_path(cls):
        return os.path.join(TestCommonClass.base_path, cls.model_name, "model", "pth_resnet50_bs1.om")

    @classmethod
    def generate_acl_json(json_path):
        # output_json_dict = json_dict
        output_json_dict = {"profiler": {"switch": "on", "aicpu": "on", "output": "testdata/profiler", "aic_metrics": ""}}
        if os.path.exists(json_path):
            shutil.rmtree(json_path)

        OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(json_path, OPEN_FLAGS, OPEN_MODES), 'w') as f:
            json.dump(output_json_dict, f, indent=4, separators=(", ", ": "), sort_keys=True)

    def init(self):
        self.model_name = "resnet50"

    def test_acl_json_using_msprof(self):
        json_path = os.path.realpath("testdata/json_file/acl_test.json")
        self.generate_acl_json(json_path)

    def test_acl_json_using_aclinit(self):
        pass

    def test_acl_json_over_size(self):
        pass

    def test_acl_json_path_not_exist(self):
        pass







if __name__ == '__main__':
    pytest.main(['test_acl_json_profiling.py', '-vs'])
