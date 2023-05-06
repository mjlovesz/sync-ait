#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2022 Huawei Technologies Co., Ltd. All rights reserved.
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
import shutil
import logging
import sys

import aclruntime
import pytest
from test_common import TestCommonClass

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass():
    def __init__(self, model_name = "no model", model_base_path = "no model path", output_file_num = -1):
        self.model_name = model_name
        self.model_base_path = model_base_path
        self.output_file_num = output_file_num

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        logger.info('\n ---class level teardown_class')

    def init(self):
        self.model_name = self.get_model_name(self)
        self.model_base_path = self.get_model_base_path(self)
        self.output_file_num = 5

    @staticmethod
    def get_model_name(self):
        return "crnn"

    def get_model_base_path(self):
        """
        supported model names as crnn, resnet50, resnet101,...。folder struct as follows
        testdata
         └── crnn   # model base
            ├── input
            ├── model
            └── output
        """
        return os.path.join(TestCommonClass.base_path, self.model_name)

    def get_dynamic_batch_om_path(self):
        return os.path.join(self.model_base_path, "model", "pth_crnn_dymbatch.om")

    def test_pure_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        batch_list = [1, 2, 4, 8, 16]

        for _, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(
                batch_size, self.model_name)
            cmd = "{} --model {} --device {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_pure_inference_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8, 16]
        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{} --model {} --device {} --dymBatch {}".format(
                TestCommonClass.cmd_prefix, model_path,
                TestCommonClass.default_device_id, dys_batch_size)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

    def test_general_inference_normal_static_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8, 16]
        base_output_path = os.path.join(self.model_base_path, "output")
        output_paths = []

        for i, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            output_batchsize_axis = 1
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {} --output_batchsize_axis {}" \
                .format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id, 
                        input_path, base_output_path, output_dirname, output_batchsize_axis)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_bin_file_num = len(os.listdir(tmp_output_path))
            assert(output_bin_file_num == self.output_file_num)
            output_paths.append(tmp_output_path)

        # compare different batchsize inference bin files
        base_compare_path = output_paths[0]
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff {}  {}".format(base_compare_path, cur_output_path)
            ret = os.system(cmd)
            assert ret == 0

        for output_path in output_paths:
            shutil.rmtree(output_path)

    def test_general_inference_normal_dynamic_batch(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)
        batch_list = [1, 2, 4, 8, 16]
        base_output_path = os.path.join(self.model_base_path, "output")
        output_paths = []

        model_path = self.get_dynamic_batch_om_path()

        for i, dys_batch_size in enumerate(batch_list):
            output_dirname = "{}_{}".format("static_batch", i)
            tmp_output_path = os.path.join(base_output_path, output_dirname)
            if os.path.exists(tmp_output_path):
                shutil.rmtree(tmp_output_path)
            os.makedirs(tmp_output_path)
            output_batchsize_axis = 1
            cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {} --output_batchsize_axis {}" \
                "--dymBatch {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                         input_path, base_output_path, output_dirname,
                                          output_batchsize_axis, dys_batch_size)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_bin_file_num = len(os.listdir(tmp_output_path))
            assert(output_bin_file_num == self.output_file_num)
            output_paths.append(tmp_output_path)

        # compare different batchsize inference bin files
        base_compare_path = output_paths[0]
        for i, cur_output_path in enumerate(output_paths):
            if i == 0:
                continue
            cmd = "diff {}  {}".format(base_compare_path, cur_output_path)
            ret = os.system(cmd)
            assert ret == 0

        for output_path in output_paths:
            shutil.rmtree(output_path)

    def test_general_inference_abnormal_output_batchsize_axis(self):
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(self.model_base_path, "input"),
                                                     self.output_file_num)

        dys_batch_size = 4
        base_output_path = os.path.join(self.model_base_path, "output")

        model_path = self.get_dynamic_batch_om_path()

        output_dirname = "{}_0".format("static_batch")
        tmp_output_path = os.path.join(base_output_path, output_dirname)
        if os.path.exists(tmp_output_path):
            shutil.rmtree(tmp_output_path)
        os.makedirs(tmp_output_path)
        output_batchsize_axis = 2
        cmd = "{} --model {} --device {} --input {} --output {} --output_dirname {} --output_batchsize_axis {}" \
            "--dymBatch {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                    input_path, base_output_path, output_dirname, output_batchsize_axis, dys_batch_size)
        logger.info("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret != 0
        shutil.rmtree(tmp_output_path)

    def test_general_inference_batchsize_is_none_normal_static_batch(self):
        batch_list = [1, 2, 4, 8, 16]
        output_parent_path = os.path.join(self.model_base_path, "output")
        output_paths = []
        summary_paths = []

        for i, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            output_dirname = "{}_{}".format("static_batch", i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            output_batchsize_axis = 1
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            cmd = "{} --model {} --device {} --output {} --output_dirname {} --output_batchsize_axis {} > {}" \
                .format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id, 
                        output_parent_path, output_dirname, output_batchsize_axis, log_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
            output_paths.append(output_path)
            summary_paths.append(summary_json_path)

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert batch_size == cur_batchsize
                    break


        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

    def test_pure_inference_batchsize_is_none_normal_dynamic_batch(self):
        batch_list = [1, 2, 4, 8, 16]
        output_parent_path = os.path.join(self.model_base_path, "output")
        output_paths = []
        summary_paths = []

        model_path = self.get_dynamic_batch_om_path()

        for i, dys_batch_size in enumerate(batch_list):
            output_dirname = "{}_{}".format("dynamic_batch", i)
            output_path = os.path.join(output_parent_path, output_dirname)
            log_path = os.path.join(output_path, "log.txt")
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
            output_batchsize_axis = 1
            summary_json_path = os.path.join(output_parent_path,  "{}_summary.json".format(output_dirname))
            cmd = "{} --model {} --device {} --output {} --output_dirname {} --output_batchsize_axis {}" \
                "--dymBatch {} > {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                             output_parent_path, output_dirname, output_batchsize_axis,
                                               dys_batch_size, log_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

            output_paths.append(output_path)
            summary_paths.append(summary_json_path)

            with open(log_path) as f:
                for line in f:
                    if "1000*batchsize" not in line:
                        continue

                    sub_str = line.split('/')[0].split('(')[1].strip(')')
                    cur_batchsize = int(sub_str)
                    assert dys_batch_size == cur_batchsize
                    break

        for output_path in output_paths:
            shutil.rmtree(output_path)
        for summary_path in summary_paths:
            os.remove(summary_path)

if __name__ == '__main__':
    pytest.main(['test_infer_crnn.py', '-vs'])






