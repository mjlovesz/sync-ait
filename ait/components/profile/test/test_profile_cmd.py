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
import pytest


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def get_cur_path(cls):
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return _current_dir

    def init(self):
        self.model_path = os.path.join(self.get_cur_path(), "testdata/resnet50/model/pth_resnet50_bs1.om")
        self.output_path = os.path.join(self.get_cur_path(), "output/")
        self.app_cmd = "ait benchmark -om {}".format(self.model_path)

    def test_default_cmd(self):
        cmd = "ait profile --application {} -o {}".format(self.app_cmd, self.output_path)
        print(cmd)
        ret = os.system(cmd)
        assert ret == 0

    def test_not_default_cmd(self):
        cmd = "ait profile --application {} -o {} --model-execution {} --sys-hardware-mem {} \
            --sys-profiling {} --sys-pid-profiling {} --dvpp-profiling {} --runtime-api {} \
            --task-time {} --aicpu {}".format(self.app_cmd, self.output_path,
                                              "on", "on", "off", "off", "on"
                                              "on", "on", "off", "off")
        print(cmd)
        ret = os.system(cmd)
        assert ret == 0

    def test_application_not_set(self):
        cmd = "ait profile -o {}".format(self.output_path)
        print(cmd)
        ret = os.system(cmd)
        assert ret != 0
