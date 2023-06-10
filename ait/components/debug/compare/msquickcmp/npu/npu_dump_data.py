#!/usr/bin/env python
# coding=utf-8
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
"""
Function:
This class mainly involves generate npu dump data function.
"""
import json
import sys
import os
import stat
import re
import shutil
import numpy as np


from msquickcmp.common import utils
from msquickcmp.common.dump_data import DumpData
from msquickcmp.common.utils import AccuracyCompareException, parse_input_shape_to_list
from msquickcmp.common.dynamic_argument_bean import DynamicArgumentEnum
from msquickcmp.npu.om_parser import OmParser

BENCHMARK_DIR = "benchmark"
ACL_JSON_PATH = "acl.json"
NPU_DUMP_DATA_BASE_PATH = "dump_data/npu"
RESULT_DIR = "result"
INPUT = "input"
INPUT_SHAPE = "--input_shape"
OUTPUT_SIZE = "--outputSize"
OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


class DynamicInput(object):

    def __init__(self, om_parser, arguments):
        self.arguments = arguments
        self.om_parser = om_parser
        self.atc_dynamic_arg, self.cur_dynamic_arg = self.get_dynamic_arg_from_om(om_parser)
        self.dynamic_arg_value = self.get_arg_value(om_parser, arguments)

    @staticmethod
    def get_dynamic_arg_from_om(om_parser):
        atc_cmd_args = om_parser.get_atc_cmdline().split(" ")
        for i, atc_arg in enumerate(atc_cmd_args):
            for dym_arg in DynamicArgumentEnum:
                if dym_arg.value.atc_arg in atc_arg:
                    if dym_arg.value.atc_arg == atc_arg:
                        atc_arg += '='+atc_cmd_args[i+1]
                    return atc_arg, dym_arg
        return "", None

    @staticmethod
    def get_input_shape_from_om(om_parser):
        # get atc input shape from atc cmdline
        atc_input_shape = ""
        atc_cmd_args = om_parser.get_atc_cmdline().split(" ")
        for i, atc_arg in atc_cmd_args:
            if INPUT_SHAPE in atc_arg:
                if INPUT_SHAPE == atc_arg:
                    atc_arg += '='+atc_cmd_args[i+1]
                atc_input_shape = atc_arg.split(utils.EQUAL)[1]
                break
        return atc_input_shape

    @staticmethod
    def get_arg_value(om_parser, arguments):
        is_dynamic_scenario, scenario = om_parser.get_dynamic_scenario_info()
        if not is_dynamic_scenario:
            utils.logger.info("The input of model is not dynamic.")
            return ""
        if om_parser.shape_range or scenario == DynamicArgumentEnum.DYM_DIMS:
            return getattr(arguments, DynamicArgumentEnum.DYM_SHAPE.value.msquickcmp_arg)
        atc_input_shape = DynamicInput.get_input_shape_from_om(om_parser)
        # from atc input shape and current input shape to get input batch size
        # if dim in shape is -1, the shape in the index of current input shape is the batch size
        atc_input_shape_dict = utils.parse_input_shape(atc_input_shape)
        quickcmp_input_shape_dict = utils.parse_input_shape(arguments.input_shape)
        batch_size_set = set()
        for op_name in atc_input_shape_dict.keys():
            DynamicInput.get_dynamic_dim_values(atc_input_shape_dict.get(op_name),
                                                quickcmp_input_shape_dict.get(op_name),
                                                batch_size_set)
        if len(batch_size_set) == 1:
            for batch_size in batch_size_set:
                return str(batch_size)
        utils.logger.error("Please check your input_shape arg is valid.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    @staticmethod
    def get_dynamic_dim_values(dym_shape, cur_shape, shape_values):
        for (dim, _) in enumerate(dym_shape):
            if dym_shape[dim] != "-1":
                continue
            if isinstance(shape_values, list):
                shape_values.append(int(cur_shape[dim]))
            else:
                shape_values.add(cur_shape[dim])

    def add_dynamic_arg_for_benchmark(self, benchmark_cmd: list):
        if self.is_dynamic_shape_scenario():
            self.check_input_dynamic_arg_valid()
            benchmark_cmd.append(self.cur_dynamic_arg.value.benchmark_arg)
            benchmark_cmd.append(self.dynamic_arg_value)

    def is_dynamic_shape_scenario(self):
        """
        if atc cmdline contain dynamic argument
        """
        return self.atc_dynamic_arg != ""

    def judge_dynamic_shape_scenario(self, atc_dym_arg):
        """
        check the dynamic shape scenario
        """
        return self.atc_dynamic_arg.split(utils.EQUAL)[0] == atc_dym_arg

    def check_input_dynamic_arg_valid(self):
        if self.cur_dynamic_arg is DynamicArgumentEnum.DYM_SHAPE:
            return
        # check dynamic input value is valid, "--arg=value" ,split by '='
        dynamic_arg_values = self.atc_dynamic_arg.split(utils.EQUAL)[1]
        if self.judge_dynamic_shape_scenario(DynamicArgumentEnum.DYM_DIMS.value.atc_arg):
            self.check_dynamic_dims_valid(dynamic_arg_values)
            return
        if self.judge_dynamic_shape_scenario(DynamicArgumentEnum.DYM_BATCH.value.atc_arg):
            self.check_dynamic_batch_valid(dynamic_arg_values)

    def check_dynamic_batch_valid(self, atc_dynamic_arg_values):
        dynamic_arg_values = atc_dynamic_arg_values.replace(utils.COMMA, utils.SEMICOLON)
        try:
            atc_value_list = utils.parse_arg_value(dynamic_arg_values)
            cur_input = utils.parse_value_by_comma(self.dynamic_arg_value)
        except AccuracyCompareException as err:
            utils.logger.error("Please input the valid shape, "
                                "the valid dynamic value range are {}".format(dynamic_arg_values))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR) from err
        for value in atc_value_list:
            if cur_input == value:
                return

    def check_dynamic_dims_valid(self, atc_dynamic_arg_values):
        atc_input_shape = DynamicInput.get_input_shape_from_om(self.om_parser)
        try:
            atc_input_shape_dict = utils.parse_input_shape(atc_input_shape)
            quickcmp_input_shape_dict = utils.parse_input_shape(self.dynamic_arg_value)
            dym_dims = []
            for op_name in atc_input_shape_dict.keys():
                DynamicInput.get_dynamic_dim_values(atc_input_shape_dict.get(op_name),
                                                    quickcmp_input_shape_dict.get(op_name),
                                                    dym_dims)
            atc_value_list = utils.parse_arg_value(atc_dynamic_arg_values)
        except AccuracyCompareException as err:
            utils.logger.error("Please input the valid shape, "
                                "the valid dynamic value range are {}".format(atc_dynamic_arg_values))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR) from err
        for value in atc_value_list:
            if dym_dims == value:
                return


class NpuDumpData(DumpData):
    """
    Class for generate npu dump data
    """

    def __init__(self, arguments, output_json_path):
        super().__init__()
        self.arguments = arguments
        self.om_parser = OmParser(output_json_path)
        self.dynamic_input = DynamicInput(self.om_parser, self.arguments)
        self.python_version = sys.executable or "python3"

    @staticmethod
    def _write_content_to_acl_json(acl_json_path, model_name, npu_data_output_dir):
        load_dict = {
            "dump": {
                "dump_list": [{"model_name": model_name}],
                "dump_path": npu_data_output_dir,
                "dump_mode": "all",
                "dump_op_switch": "off"
            }
        }
        if os.access(acl_json_path, os.W_OK):
            try:
                with os.fdopen(os.open(acl_json_path, OPEN_FLAGS, OPEN_MODES), "w") as write_json:
                    try:
                        json.dump(load_dict, write_json)
                    except ValueError as exc:
                        utils.logger.info(str(exc))
                        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_WRITE_JSON_FILE_ERROR) from exc
            except IOError as acl_json_file_except:
                utils.logger.error('Failed to open"' + acl_json_path + '", ' + str(acl_json_file_except))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from acl_json_file_except
        else:
            utils.logger.error(
                "The path {} does not have permission to write.Please check the path permission".format(acl_json_path))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)

    def generate_inputs_data(self):
        if self.arguments.input_path:
            input_path = self.arguments.input_path.split(",")
            for i, input_file in enumerate(input_path):
                if not os.path.isfile(input_file):
                    utils.logger.error("no such file exists: {}".format(input_file))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                file_name = "input_" + str(i) + ".bin"
                dest_file = os.path.join(self.arguments.out_path, "input", file_name)
                shutil.copy(input_file, dest_file)
            return
        aipp_content = self.om_parser.get_aipp_config_content()
        aipp_list = aipp_content.split(",")
        src_image_size_h = []
        src_image_size_w = []
        for aipp_info in aipp_list:
            if "src_image_size_h" in aipp_info:
                src_image_size_h.append(aipp_info.split(":")[1])
            if "src_image_size_w" in aipp_info:
                src_image_size_w.append(aipp_info.split(":")[1])
        if not src_image_size_h or not src_image_size_w:
            utils.logger.error("atc insert_op_config file contains no src_image_size_h or src_image_size_w")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT)
        if len(src_image_size_h) != len(src_image_size_w):
            utils.logger.error("atc insert_op_config file's src_image_size_h number "
                                  "does not equal src_image_size_w")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT)
        if self.arguments.input_shape:
            inputs_list = parse_input_shape_to_list(self.arguments.input_shape)
        else:
            inputs_list = self.om_parser.get_shape_list()
        if len(inputs_list) != len(src_image_size_h):
            utils.logger.error("inputs number is not equal to aipp inputs number, please check the -s param")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_WRONG_AIPP_CONTENT)
        # currently, onnx only support input format nchw
        h_position = 2
        w_position = 3
        input_dir = os.path.join(self.arguments.out_path, "input")
        for i, item in enumerate(inputs_list):
            item[h_position] = int(src_image_size_h[i])
            item[w_position] = int(src_image_size_w[i])
            input_data = np.random.randint(0, 256, item).astype(np.uint8)
            file_name = "input_" + str(i) + ".bin"
            input_data.tofile(os.path.join(input_dir, file_name))

    def generate_dump_data(self, use_cli):
        """
        Function Description:
            compile and rum benchmark project
        Return Value:
            npu dump data path
        """
        self._check_input_path_param()
        if not use_cli:
            benchmark_dir = os.path.join(os.path.realpath("../../"), BENCHMARK_DIR)
            self.benchmark_install_sh(benchmark_dir)
        return self.benchmark_run()

    def get_expect_output_name(self):
        """
        Function Description:
            get expect output node name in golden net
        Return Value:
            output node name in golden net
        """
        return self.om_parser.get_expect_net_output_name()

    def benchmark_install_sh(self, benchmark_dir):
        """
        Function Description:
            compile benchmark backend project
        Parameter:
            benchmark_dir: benchmark project directory
        """
        execute_path = benchmark_dir
        utils.logger.info("Start to install benchmark backend execute_path: %s" % execute_path)
        install_sh_cmd = ["sh", "install.sh", "-p", self.python_version]

        retval = os.getcwd()
        os.chdir(execute_path)

        # do install.sh command
        utils.logger.info("Run command line: cd %s && %s" % (execute_path, " ".join(install_sh_cmd)))
        utils.execute_command(install_sh_cmd)
        utils.logger.info("Finish to install benchmark backend execute_path: %s." % benchmark_dir)
        os.chdir(retval)
        utils.logger.info("Run command line: cd %s (back to the working directory)" % (retval))

    def benchmark_run(self):
        """
        Function Description:
            run benchmark project
        Return Value:
            npu dump data path
        Exception Description:
            when invalid npu dump data path throw exception
        """
        try:
            import ais_bench
        except ModuleNotFoundError as err:
            raise err

        self._compare_shape_vs_file()
        npu_data_output_dir = os.path.join(self.arguments.out_path, NPU_DUMP_DATA_BASE_PATH)
        utils.create_directory(npu_data_output_dir)
        model_name, extension = utils.get_model_name_and_extension(self.arguments.offline_model_path)
        acl_json_path = os.path.join(npu_data_output_dir, ACL_JSON_PATH)
        if not os.path.exists(acl_json_path):
            os.mknod(acl_json_path, mode=0o600)
        benchmark_cmd = [self.python_version, "-m", "ais_bench", "--model", self.arguments.offline_model_path,
                         "--input", self.arguments.benchmark_input_path, "--device", self.arguments.device,
                         "--output", npu_data_output_dir]
        if self.arguments.dump:
            cur_dir = os.getcwd()
            acl_json_path = os.path.join(cur_dir, acl_json_path)
            self._write_content_to_acl_json(acl_json_path, model_name, npu_data_output_dir)
            benchmark_cmd.extend(["--acl_json_path", acl_json_path])

        self.dynamic_input.add_dynamic_arg_for_benchmark(benchmark_cmd)
        self._make_benchmark_cmd_for_shape_range(benchmark_cmd)

        # do benchmark command
        utils.logger.info("Run command line: %s" % (benchmark_cmd))
        utils.execute_command(benchmark_cmd)

        npu_dump_data_path = ""
        if self.arguments.dump:
            npu_dump_data_path, file_is_exist = utils.get_dump_data_path(npu_data_output_dir)
            if not file_is_exist:
                utils.logger.error("The path {} dump data is not exist.".format(npu_dump_data_path))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        # net output data path
        npu_net_output_data_path, file_is_exist = utils.get_dump_data_path(npu_data_output_dir, True)
        if not file_is_exist:
            utils.logger.error("The path {} net output data is not exist.".format(npu_net_output_data_path))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        self._convert_net_output_to_numpy(npu_net_output_data_path, npu_dump_data_path)
        return npu_dump_data_path, npu_net_output_data_path

    def _make_benchmark_cmd_for_shape_range(self, benchmark_cmd):
        pattern = re.compile(r'^[0-9]+$')
        count = self.om_parser.get_net_output_count()
        if not self.arguments.output_size:
            if count > 0:
                count_list = []
                for _ in range(count):
                    count_list.append("90000000")
                self.arguments.output_size = ",".join(count_list)
        if self.arguments.output_size:
            output_size_list = self.arguments.output_size.split(',')
            if len(output_size_list) != count:
                utils.logger.error(
                    'The output size (%d) is not equal %d in model. Please check the "--output-size" argument.'
                    % (len(output_size_list), count))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            for item in output_size_list:
                item = item.strip()
                match = pattern.match(item)
                if match is None:
                    utils.logger.error("The size (%s) is invalid. Please check the output size."
                                          % self.arguments.output_size)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
                if int(item) <= 0:
                    utils.logger.error("The size (%s) must be large than zero. Please check the output size."
                                          % self.arguments.output_size)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            benchmark_cmd.append(OUTPUT_SIZE)
            benchmark_cmd.append(self.arguments.output_size)

    def _convert_net_output_to_numpy(self, npu_net_output_data_path, npu_dump_data_path):
        net_output_data = None
        npu_net_output_data_info = self.om_parser.get_net_output_data_info(npu_dump_data_path)
        for dir_path, _, files in os.walk(npu_net_output_data_path):
            for index, each_file in enumerate(sorted(files)):
                data_type = npu_net_output_data_info.get(index)[0]
                shape = npu_net_output_data_info.get(index)[1]
                data_len = utils.get_data_len_by_shape(shape)
                original_net_output_data = np.fromfile(os.path.join(dir_path, each_file), data_type, data_len)
                try:
                    net_output_data = original_net_output_data.reshape(shape)
                except ValueError:
                    utils.logger.warning(
                        "The shape of net_output data from file {} is {}.".format(
                            each_file, shape))
                    net_output_data = original_net_output_data
                file_name = os.path.basename(each_file).split('.')[0]
                numpy_file_path = os.path.join(npu_net_output_data_path, file_name)
                utils.save_numpy_data(numpy_file_path, net_output_data)

    def _check_input_path_param(self):
        if self.arguments.input_path == "":
            input_path = os.path.join(self.arguments.out_path, INPUT)
            utils.check_file_or_directory_path(os.path.realpath(input_path), True)
            input_bin_files = os.listdir(input_path)
            input_bin_files.sort(key=lambda file: int((re.findall("\\d+", file))[0]))
            bin_file_path_array = []
            for item in input_bin_files:
                bin_file_path_array.append(os.path.join(input_path, item))
            self.arguments.benchmark_input_path = ",".join(bin_file_path_array)
        else:
            bin_file_path_array = utils.check_input_bin_file_path(self.arguments.input_path)
            self.arguments.benchmark_input_path = ",".join(bin_file_path_array)

    def _compare_shape_vs_file(self):
        shape_size_array = self.om_parser.get_shape_size()
        if self.om_parser.contain_negative_1:
            return
        files_size_array = self._get_file_size()
        self._shape_size_vs_file_size(shape_size_array, files_size_array)

    def _get_file_size(self):
        file_size = []
        files = self.arguments.benchmark_input_path.split(",")
        for item in files:
            if item.endswith("bin") or item.endswith("BIN"):
                file_size.append(os.path.getsize(item))
            elif item.endswith("npy") or item.endswith("NPY"):
                try:
                    file_size.append(np.load(item).size)
                except (ValueError, FileNotFoundError) as e:
                    utils.logger.error("The path {} can not get its size through numpy".format(item))
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR) from e
            else:
                utils.logger.error("Input_path parameter only support bin or npy file, "
                                      "but got {}".format(item))
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        return file_size

    def _shape_size_vs_file_size(self, shape_size_array, files_size_array):
        if len(shape_size_array) < len(files_size_array):
            utils.logger.error("The number of input files is incorrect.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        if self.om_parser.shape_range:
            for file_size in files_size_array:
                if file_size not in shape_size_array:
                    utils.logger.error(
                        "The size (%d) of file can not match the input of the model." % file_size)
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
        elif self.dynamic_input.is_dynamic_shape_scenario():
            for shape_size in shape_size_array:
                for file_size in files_size_array:
                    if file_size <= shape_size:
                        return
            utils.logger.warning("The size of bin file can not match the input of the model.")
        else:
            for shape_size, file_size in zip(shape_size_array, files_size_array):
                if shape_size == 0:
                    continue
                if shape_size != file_size:
                    utils.logger.error("The shape value is different from the size of the file.")
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_BIN_FILE_ERROR)
