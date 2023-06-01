# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
import logging
import subprocess
import sys
import os
import pytest
from msquickcmp.cmp_process import run
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.atc.atc_utils import AtcUtils
from msquickcmp.common import utils
from msquickcmp.accuracy_locat.accuracy_locat import find_accuracy_interval

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    # staticmethod or classmethod
    @classmethod
    def get_base_path(cls):
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(_current_dir, "../tests")

    @classmethod
    def get_cann_path(cls):
        result = subprocess.run(['which', 'atc'], stdout=subprocess.PIPE)
        atc_path = result.stdout.decode('utf-8').strip()
        cann_path = atc_path[:-8]
        return cann_path

    @classmethod
    def set_accumulate_cmp_args(cls):
        args_data2vec_cmp = CmpArgsAdapter(
            os.path.join(cls.get_base_path(), 'onnx/data2vec_1_108.onnx'), # gold_model
            os.path.join(cls.get_base_path(), 'om/data2vec_1_108.om'), # om_model
            "{},{}".format(
                os.path.join(cls.get_base_path(), 'input_datas/data2vec/1535_0'),
                os.path.join(cls.get_base_path(), 'input_datas/data2vec/1535_1')
            ), # input_data_path
            cls.cann_path, # cann_path
            os.path.join(cls.get_base_path(), '/test/data2vec/output/'), # out_path
            "", # input_shape
            "0", # device
            "", # output_size
            "", # output_nodes
            False, # advisor
            "", # dym_shape_range
            True, # dump
            False # bin2npy
        )
        return args_data2vec_cmp

    @classmethod
    def check_and_run(cls, args:CmpArgsAdapter, use_cli:bool):
        utils.check_file_or_directory_path(args.model_path)
        utils.check_file_or_directory_path(args.offline_model_path)
        utils.check_device_param_valid(args.device)
        utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
        utils.check_convert_is_valid_used(args.dump, args.bin2npy)

        original_out_path = os.path.realpath(os.path.join(args.out_path, "log/"))
        args.out_path = original_out_path

        # convert the om model to json
        output_json_path = AtcUtils(args).convert_model_to_json()

        # deal with the dymShape_range param if exists
        input_shapes = []
        if args.dym_shape_range:
            input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
        if not input_shapes:
            input_shapes.append("")
        for input_shape in input_shapes:
            res = run(args, input_shape, output_json_path, original_out_path, use_cli)

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.cann_path = self.get_cann_path()
        self.args_data2vec_cmp = self.set_accumulate_cmp_args
        self.check_and_run(self.args_data2vec_cmp, True)


        self.args_data2vec_acc = CmpArgsAdapter(
            os.path.join(self.get_base_path(), 'onnx/data2vec_1_108.onnx'), # gold_model
            os.path.join(self.get_base_path(), 'om/data2vec_1_108.om'), # om_model
            "", # input_data_path
            self.cann_path, # cann_path
            os.path.join(self.get_base_path(), '/test/data2vec/output/log/'), # out_path
            "", # input_shape
            "0", # device
            "", # output_size
            "", # output_nodes
            False, # advisor
            "", # dym_shape_range
            True, # dump
            False # bin2npy
        )

 # testcases

    def test_compare_accumlate_accuracy_situation(self):
        logger.info(self.args_data2vec_acc.model_path)
        logger.info(self.args_data2vec_acc.offline_model_path)
        logger.info(self.args_data2vec_acc.input_path)
        logger.info(self.args_data2vec_acc.cann_path)
        logger.info(self.args_data2vec_acc.out_path)
        logger.info(self.args_data2vec_acc.input_shape)
        logger.info(self.args_data2vec_acc.device)
        logger.info(self.args_data2vec_acc.output_size)
        logger.info(self.args_data2vec_acc.output_nodes)
        logger.info(self.args_data2vec_acc.advisor)
        logger.info(self.args_data2vec_acc.dym_shape_range)
        logger.info(self.args_data2vec_acc.dump)
        logger.info(self.args_data2vec_acc.bin2npy)
        find_accuracy_interval(self.args_data2vec_acc, "Gather_1186", "")
