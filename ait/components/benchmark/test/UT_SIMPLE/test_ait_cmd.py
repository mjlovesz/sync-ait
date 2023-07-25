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
import sys
import logging
import argparse

import numpy as np
import pytest
from test_common import TestCommonClass
from ais_bench.infer.main_cli import BenchmarkCommand

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @staticmethod
    @pytest.fixture
    def cmdline_legal_args(self, monkeypatch):
        monkeypatch.setattr('sys.argv', self.case_cmd_list)

    @classmethod
    def cmd_dict_to_list(cls, cmd_dict):
        cmd_list = ['test_ait_cmd.py']
        for key, value in cmd_dict.items():
            cmd_list.append(key)
            cmd_list.append(value)
        return cmd_list

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.model_name = "resnet50"
        self.benchmark_command = BenchmarkCommand()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_cmd_dict = {
            "--om-model": os.path.join(self.current_dir, "../testdata/resnet50/model/pth_resnet50_bs4.om"),
            "--input": "datasets/",
            "--output": "output/"
        }
        # self.base_cmd_dict = {
        #     "--om-model": os.path.join(self.current_dir, "../testdata/resnet50/model/pth_resnet50_bs4.om"),
        #     "--input": "datasets/",
        #     "--output": "output/",
        #     "--output-dirname": "outdir/",
        #     "--outfmt": "NPY",
        #     "--loop": "100",
        #     "--debug": "0",
        #     "--device": "0,1",
        #     "--dym-batch": "16",
        #     "--dym-hw": "224,224",
        #     "--dym-dims": "1,3,224,224",
        #     "--dym-shape": "1,3,224,224",
        #     "--output-size": "10000",
        #     "--auto-set-dymshape-mode": "0",
        #     "--auto-set-dymdims-mode": "0",
        #     "--batch-size": "16",
        #     "--pure-data-type": "zero",
        #     "--profiler": "0",
        #     "--dump": "0",
        #     "--acl-json-path": "acl.json",
        #     "--output-batchsize-axis": "1",
        #     "--run-mode": "array",
        #     "--display-all-summary": "0",
        #     "--warmup-count": "1",
        #     "--dym-shape-range": "1~3,3,224,224-226",
        #     "--aipp-config": "aipp.config",
        #     "--energy_consumption": "0",
        #     "--npu_id": "0",
        #     "--backend": "trtexec",
        #     "--perf": "0",
        #     "--pipeline": "0",
        #     "--profiler-rename": "0",
        #     "--dump-npy": "0"
        # }
        self.case_cmd_list = []

    def test_check_all_full_args_legality(self):
        cmd_dict = self.base_cmd_dict
        self.case_cmd_list = self.cmd_dict_to_list(cmd_dict)
        def run(func):
            parser = argparse.ArgumentParser()
            self.benchmark_command.add_arguments(parser)
            args = parser.parse_args()
            assert args.input == "datasets/"
        run(self.cmdline_legal_args)




