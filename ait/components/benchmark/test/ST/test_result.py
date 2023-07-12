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
import sys
import logging
import subprocess

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

    def init(self):
        self.model_name = "resnet50"

    def get_dynamic_batch_om_path(self):
        return os.path.join(TestCommonClass.base_path, self.model_name, "model", "pth_resnet50_dymbatch.om")

    def test_args_ok(self):
        output_path = os.path.join(TestCommonClass.base_path, "tmp")
        TestCommonClass.prepare_dir(output_path)
        model_path = TestCommonClass.get_model_static_om_path(2, self.model_name)
        cmd = "{} --model {} --device {}".format(TestCommonClass.cmd_prefix, model_path,
                                                 TestCommonClass.default_device_id)
        cmd = "{} --output {}".format(cmd, output_path)
        logger.info("run cmd:{}".format(cmd))
        ret = os.system(cmd)
        assert ret == 0

    def test_general_inference_normal_static_batch(self):
        """
        batch size 1,2,4,8
        """
        warmup_num = 1
        output_file_num = 17
        batch_list = [1, 2, 4, 8, 16]
        output_path = os.path.join(TestCommonClass.base_path, self.model_name, "output")
        TestCommonClass.prepare_dir(output_path)
        log_path = os.path.join(output_path, "log.txt")
        result_paths = []
        summary_json_paths = []
        batch_size = 1
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(os.path.join(TestCommonClass.base_path,
                                                                                           self.model_name), "input"),
                                                     output_file_num)

        for _, batch_size in enumerate(batch_list):
            model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
            cmd = "{} --model {} --device {} --output {} --debug True \
                --input {} > {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                        output_path, input_path, log_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

            # inference times should be  fit to given rule
            real_execute_num = TestCommonClass.get_inference_execute_num(log_path)
            if batch_size != 0:
                exacute_num = math.ceil(output_file_num/batch_size)
                assert real_execute_num == warmup_num + exacute_num
            else:
                logger.warning("zero division!")

            # bin file num is equal to output_file_num
            cmd = "cat {} |grep 'output path'".format(log_path)
            try:
                outval = os.popen(cmd).read()
            except Exception as e:
                raise Exception("grep action raises an exception: {}".format(e)) from e

            result_path = os.path.join(output_path, outval.split(':')[1].replace('\n', ''))
            result_paths.append(result_path)
            summary_json_name = result_path.split("/")[-1]
            summary_json_paths.append(os.path.join(output_path, "{}_summary.json".format(summary_json_name)))

        # bin file compare for batch size [1, 2,4,8], should be same
        for _, result_path in enumerate(result_paths[1:]):
            cmd = "diff  {}  {}".format(result_paths[0], result_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0
        for output_dir_path in result_paths:
            shutil.rmtree(output_dir_path)
        for summary_json_path in summary_json_paths:
            os.remove(summary_json_path)

    def test_pipeline_inference_normal_static_batch(self):
        warmup_num = 5
        output_file_num = 20
        general_output_path = os.path.join(TestCommonClass.base_path, self.model_name, "output_general")
        pipeline_output_path = os.path.join(TestCommonClass.base_path, self.model_name, "output_pipeline")
        TestCommonClass.prepare_dir(general_output_path)
        TestCommonClass.prepare_dir(pipeline_output_path)
        general_log_path = os.path.join(general_output_path, "log.txt")
        pipeline_log_path = os.path.join(pipeline_output_path, "log.txt")
        run_modes = {"general" : {"output_path" : general_output_path, "log_path" : general_log_path,
                                  "results_paths" : [], "summary_json_paths" : []},
                     "pipeline" : {"output_path" : pipeline_output_path, "log_path" : pipeline_log_path,
                                  "results_paths" : [], "summary_json_paths" : []}}
        batch_size = 1
        model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(os.path.join(TestCommonClass.base_path,
                                                                                           self.model_name), "input"),
                                                     output_file_num)

        for mode, info in run_modes.items():
            output_path = info.get("output_path")
            log_path = info.get("log_path")
            cmd = "{} --model {} --device {} --output {} --debug True \
                --input {} > {}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                        output_path, input_path, log_path)
            logger.info("run in {} mode. cmd:{}".format(mode, cmd))
            ret = os.system(cmd)
            assert ret == 0

            # inference times should be  fit to given rule
            real_execute_num = TestCommonClass.get_inference_execute_num(log_path)
            if batch_size != 0:
                exacute_num = math.ceil(output_file_num/batch_size)
                assert real_execute_num == warmup_num + exacute_num
            else:
                logger.warning("zero division!")

            # bin file num is equal to output_file_num
            cmd = "cat {} |grep 'output path'".format(log_path)
            try:
                outval = os.popen(cmd).read()
            except Exception as e:
                raise Exception("grep action raises an exception: {}".format(e)) from e

            result_path = os.path.join(output_path, outval.split(':')[1].replace('\n', ''))
            info.get("results_paths").append(result_path)
            summary_json_name = result_path.split("/")[-1]
            info.get("summary_json_paths").append(os.path.join(output_path, "{}_summary.json".format(summary_json_name)))

        # compare e2e time
        cmd_general = "cat {} | grep 'end_to_end' | cut -d: -f2".format(run_modes["general"]["log_path"])
        cmd_pipeline = "cat {} | grep 'end_to_end' | cut -d: -f2".format(run_modes["pipeline"]["log_path"])
        res_general = subprocess.Popen(cmd_general, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res_pipeline = subprocess.Popen(cmd_pipeline, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        e2e_time_general, _ = res_general.communicate()
        e2e_time_pipeline, _ = res_pipeline.communicate()
        assert float(e2e_time_general) > float(e2e_time_pipeline)

        # delete tmp file
        for output_dir_path in run_modes["general"]["results_paths"] + run_modes["pipeline"]["results_paths"]:
            shutil.rmtree(output_dir_path)
        for summary_json_path in run_modes["general"]["summary_json_paths"] + run_modes["pipeline"]["summary_json_paths"]:
            os.remove(summary_json_path)

    def test_general_inference_normal_dynamic_batch(self):
        batch_size = 1
        warmup_num = 1
        output_file_num = 17
        result_paths = []
        summary_json_paths = []
        output_path = os.path.join(TestCommonClass.base_path, self.model_name, "output")
        TestCommonClass.prepare_dir(output_path)
        log_path = os.path.join(output_path, "log.txt")
        static_model_path = TestCommonClass.get_model_static_om_path(batch_size, self.model_name)
        input_size = TestCommonClass.get_model_inputs_size(static_model_path)[0]
        input_path = TestCommonClass.get_inputs_path(input_size, os.path.join(os.path.join(TestCommonClass.base_path,
                                                                                           self.model_name), "input"),
                                                     output_file_num)
        batch_list = [1, 2, 4, 8]

        model_path = self.get_dynamic_batch_om_path()
        for _, dys_batch_size in enumerate(batch_list):
            cmd = "{0} --model {1} --device {2} --debug True --dymBatch {3} --input {4} \
                --output {5} > {6}".format(TestCommonClass.cmd_prefix, model_path, TestCommonClass.default_device_id,
                                           dys_batch_size, input_path, output_path, log_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

            # inference times should be  fit to given rule
            real_execute_num = TestCommonClass.get_inference_execute_num(log_path)
            if dys_batch_size != 0:
                exacute_num = math.ceil(output_file_num/dys_batch_size)
                assert real_execute_num == warmup_num + exacute_num
            else:
                logger.warning("zero division!")

            # bin file num is equal to output_file_num
            cmd = "cat {} |grep 'output path'".format(log_path)
            try:
                outval = os.popen(cmd).read()
            except Exception as e:
                raise Exception("grep action raises an exception: {}".format(e)) from e

            result_path = os.path.join(output_path, outval.split(':')[1].replace('\n', ''))
            result_paths.append(result_path)
            summary_json_name = result_path.split("/")[-1]
            summary_json_paths.append(os.path.join(output_path, "{}_summary.json".format(summary_json_name)))

        # bin file compare for batch size [1, 2,4,8], should be same
        for _, result_path in enumerate(result_paths[1:]):
            cmd = "diff {}  {}".format(result_paths[0], result_path)
            logger.info("run cmd:{}".format(cmd))
            ret = os.system(cmd)
            assert ret == 0

        for output_dir_path in result_paths:
            shutil.rmtree(output_dir_path)
        for summary_json_path in summary_json_paths:
            os.remove(summary_json_path)


if __name__ == '__main__':
    pytest.main(['test_result.py', '-vs'])
