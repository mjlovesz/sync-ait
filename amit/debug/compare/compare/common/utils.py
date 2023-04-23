#!/usr/bin/env python
# coding=utf-8
"""
Function:
This class mainly involves common function.
Copyright Information:
HuaWei Technologies Co.,Ltd. All Rights Reserved © 2021
"""
import os
import re
import shutil
import subprocess
import sys
import time
import enum
import itertools
import argparse

import numpy as np

from compare.common.dynamic_argument_bean import DynamicArgumentEnum

ACCURACY_COMPARISON_INVALID_PARAM_ERROR = 1
ACCURACY_COMPARISON_INVALID_DATA_ERROR = 2
ACCURACY_COMPARISON_INVALID_PATH_ERROR = 3
ACCURACY_COMPARISON_INVALID_COMMAND_ERROR = 4
ACCURACY_COMPARISON_PYTHON_VERSION_ERROR = 5
ACCURACY_COMPARISON_MODEL_TYPE_ERROR = 6
ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR = 7
ACCURACY_COMPARISON_WRITE_JSON_FILE_ERROR = 8
ACCURACY_COMPARISON_OPEN_FILE_ERROR = 9
ACCURACY_COMPARISON_BIN_FILE_ERROR = 10
ACCURACY_COMPARISON_INVALID_KEY_ERROR = 11
ACCURACY_COMPARISON_PYTHON_COMMAND_ERROR = 12
ACCURACY_COMPARISON_TENSOR_TYPE_ERROR = 13
ACCURACY_COMPARISON_NO_DUMP_FILE_ERROR = 14
ACCURACY_COMPARISON_NOT_SUPPORT_ERROR = 15
ACCURACY_COMPARISON_NET_OUTPUT_ERROR = 16
ACCURACY_COMPARISON_INVALID_DEVICE_ERROR = 17
MODEL_TYPE = ['.onnx', '.pb', '.om']
DIM_PATTERN = r"^(-?[0-9]+)(,-?[0-9]+)*"
DYNAMIC_DIM_PATTERN = r"^([0-9-~]+)(,-?[0-9-~]+){0,3}"
MAX_DEVICE_ID = 255
SEMICOLON = ";"
COLON = ":"
EQUAL = "="
COMMA = ","
DOT = "."
ASCEND_BATCH_FIELD = "ascend_mbatch_batch_"
BATCH_SCENARIO_OP_NAME = "{0}_ascend_mbatch_batch_{1}"

class AccuracyCompareException(Exception):
    """
    Class for Accuracy Compare Exception
    """

    def __init__(self, error_info):
        super(AccuracyCompareException, self).__init__()
        self.error_info = error_info


class InputShapeError(enum.Enum):
    """
     Class for Input Shape Error
    """
    FORMAT_NOT_MATCH = 0
    VALUE_TYPE_NOT_MATCH = 1
    NAME_NOT_MATCH = 2
    TOO_LONG_PARAMS = 3


def _print_log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg)
    sys.stdout.flush()


def print_info_log(info_msg):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg)


def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warn message.
    """
    _print_log("WARNING", warn_msg)


def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """

    if isdir:
        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.Please check the path'.format(path))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        if not os.access(path, os.W_OK):
            print_error_log(
                'The path{} does not have permission to write.Please check the path permission'.format(path))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('The path {} is not a file.Please check the path'.format(path))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        if not os.access(path, os.R_OK):
            print_error_log(
                'The path{} does not have permission to read.Please check the path permission'.format(path))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)


def get_model_name_and_extension(offline_model_path):
    """
    Function Description:
        obtain the name and extension of the model file
    Parameter:
        offline_model_path: offline model path
    Return Value:
        model_name,extension
    Exception Description:
        when invalid data throw exception
    """
    file_name = os.path.basename(offline_model_path)
    model_name, extension = os.path.splitext(file_name)
    if extension not in MODEL_TYPE:
        print_error_log('Only model files whose names end with .pb or .onnx are supported.Please check {}'.format(
            offline_model_path))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)
    return model_name, extension


def get_dump_data_path(dump_dir, is_net_output=False):
    """
    Function Description:
        traverse directories and obtain the absolute path of dump data
    Parameter:
        dump_dir: dump data directory
    Return Value:
        dump data path,file is exist or file is not exist
    """
    dump_data_path = None
    file_is_exist = False
    dump_data_dir = None
    for i in os.listdir(dump_dir):
        if not (os.path.isdir(os.path.join(dump_dir, i))):
            continue
        # net_output dump file directory, name is like 12_423_246_4352
        if is_net_output:
            if not i.isdigit():
                dump_data_dir = os.path.join(dump_dir, i)
                break
        # Contains the dump file directory, whose name is a pure digital timestamp
        elif i.isdigit():
            dump_data_dir = os.path.join(dump_dir, i)
            break

    if not dump_data_dir:
        print_error_log("The directory \"{}\" does not contain dump data".format(dump_dir))
        raise AccuracyCompareException(ACCURACY_COMPARISON_NO_DUMP_FILE_ERROR)

    for dir_path, sub_paths, files in os.walk(dump_data_dir):
        if len(files) != 0:
            dump_data_path = dir_path
            file_is_exist = True
            break
        dump_data_path = dir_path
    return dump_data_path, file_is_exist


def execute_command(cmd):
    """
    Function Description:
        run the following command
    Parameter:
        cmd: command
    Return Value:
        command output result
    Exception Description:
        when invalid command throw exception
    """
    print_info_log('Execute command:%s' % cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            print(line)
    if process.returncode != 0:
        print_error_log('Failed to execute command:%s' % " ".join(cmd))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DATA_ERROR)


def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}.Please check the path permission or disk space .{}'.format(dir_path, str(ex)))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PATH_ERROR)


def check_input_bin_file_path(input_path):
    """
    Function Description:
        check the output bin file
    Parameter:
        input_path: input path directory
    """
    input_bin_files = input_path.split(',')
    bin_file_path_array = []
    for input_item in input_bin_files:
        input_item_path = os.path.realpath(input_item)
        check_file_or_directory_path(input_item_path)
        bin_file_path_array.append(input_item_path)
    return bin_file_path_array


def check_dynamic_shape(shape):
    """
    Function Description:
        check dynamic shpae
    Parameter:
        shape:shape
    Return Value:
        False or True
    """
    dynamic_shape = False
    for item in shape:
        if item is None or isinstance(item, str):
            dynamic_shape = True
            break
    return dynamic_shape


def parse_input_shape(input_shape):
    """
    Function Description:
        parse input shape
    Parameter:
        input_shape:the input shape,this format like:tensor_name1:dim1,dim2;tensor_name2:dim1,dim2
    Return Value:
        the map type of input_shapes
    """
    input_shapes = {}
    if input_shape == '':
        return input_shapes
    _check_colon_exist(input_shape)
    tensor_list = input_shape.split(';')
    for tensor in tensor_list:
        _check_colon_exist(input_shape)
        tensor_shape_list = tensor.rsplit(':', maxsplit=1)
        if len(tensor_shape_list) == 2:
            shape = tensor_shape_list[1]
            input_shapes[tensor_shape_list[0]] = shape.split(',')
            _check_shape_number(shape)
        else:
            print_error_log(get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, input_shape))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return input_shapes


def parse_dymshape_range(dymshape_range):
    _check_colon_exist(dymshape_range)
    input_shapes = {}
    tensor_list = dymshape_range.split(";")
    info_list = []
    for tensor in tensor_list:
        _check_colon_exist(dymshape_range)
        shapes = []
        name, shapestr = tensor.split(":")
        if len(shapestr) < 50:
            _check_shape_number(shapestr, DYNAMIC_DIM_PATTERN)
        else:
            print_error_log(get_shape_not_match_message(InputShapeError.TOO_LONG_PARAMS, input_shape))
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        for content in shapestr.split(","):
            if "~" in content:
                content_split = content.split("~")
                _check_content_split_length(content_split)
                start = int(content_split[0])
                end = int(content_split[1])
                step = int(content_split[2]) if len(content_split) == 3 else 1
                ranges = [str(i) for i in range(start, end+1, step)]
            elif "-" in content:
                ranges = content.split("-")
            else:
                start = int(content)
                ranges = [str(start)]
            shapes.append(ranges)
        shape_list = [",".join(s) for s in list(itertools.product(*shapes))]
        info = ["{}:{}".format(name, s) for s in shape_list]
        info_list.append(info)
    res = [";".join(s) for s in list(itertools.product(*info_list))]
    print_info_log("shape_list:" + str(res))
    return res


def get_shape_to_directory_name(input_shape):
    shape_info = re.sub(r"[:;]", "-", input_shape)
    shape_info = re.sub(r",", "_", shape_info)
    return shape_info


def _check_colon_exist(input_shape):
    if ":" not in input_shape:
        print_error_log(get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, input_shape))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def _check_content_split_length(content_split):
    if not content_split[1]:
        print_error_log(get_shape_not_match_message(InputShapeError.VALUE_TYPE_NOT_MATCH, content_split[1]))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def _check_shape_number(input_shape_value, pattern=DIM_PATTERN):
    dim_pattern = re.compile(pattern)
    match = dim_pattern.match(input_shape_value)
    if not match or match.group() is not input_shape_value:
        print_error_log(get_shape_not_match_message(InputShapeError.VALUE_TYPE_NOT_MATCH, input_shape_value))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def get_shape_not_match_message(shape_error_type, value):
    """
    Function Description:
        get shape not match message
    Parameter:
        input:the value
        shape_error_type: the shape error type
    Return Value:
        not match message
    """
    message = ""
    if shape_error_type == InputShapeError.FORMAT_NOT_MATCH:
        message = "Input shape \"{}\" format mismatch,the format like: " \
                  "input_name1:1,224,224,3;input_name2:3,300".format(value)
    if shape_error_type == InputShapeError.VALUE_TYPE_NOT_MATCH:
        message = "Input shape \"{}\" value not number".format(value)
    if shape_error_type == InputShapeError.NAME_NOT_MATCH:
        message = "Input tensor name \"{}\" not in model".format(value)
    if shape_error_type == InputShapeError.TOO_LONG_PARAMS:
        message = "Input \"{}\" value too long".format(value)
    return message


def check_input_name_in_model(tensor_name_list, input_name):
    """
    Function Description:
        check input name in model
    Parameter:
        tensor_name_list: the tensor name list
        input_name: the input name
    Exception Description:
        When input name not in tensor name list throw exception
    """
    if input_name not in tensor_name_list:
        print_error_log(get_shape_not_match_message(InputShapeError.NAME_NOT_MATCH, input_name))
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)


def save_numpy_data(file_path, data):
    """
    save_numpy_data
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)


def check_device_param_valid(device):
    """
    check device param valid.
    """
    if not device.isdigit() or int(device) > MAX_DEVICE_ID:
        print_error_log(
            "Please enter a valid number for device, the device id should be"
            " in [0, 255], now is %s." % device)
        raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_DEVICE_ERROR)


def parse_arg_value(values):
    """
    parse dynamic arg value of atc cmdline
    """
    value_list = []
    for item in values.split(SEMICOLON):
        value_list.append(parse_value_by_comma(item))
    return value_list


def parse_value_by_comma(value):
    """
    parse value by comma, like '1,2,4,8'
    """
    value_list = []
    value_str_list = value.split(COMMA)
    for value_str in value_str_list:
        value_str = value_str.strip()
        if value_str.isdigit() or value_str == '-1':
            value_list.append(int(value_str))
        else:
            print_error_log("please check your input shape.")
            raise AccuracyCompareException(ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
    return value_list


def get_batch_index(dump_data_path):
    for _, _, files in os.walk(dump_data_path):
        for file_name in files:
            if ASCEND_BATCH_FIELD in file_name:
                return get_batch_index_from_name(file_name)
    return ""


def handle_ground_truth_files(om_parser, npu_dump_data_path, golden_dump_data_path):
    _, scenario = om_parser.get_dynamic_scenario_info()
    if scenario in [DynamicArgumentEnum.DYM_BATCH, DynamicArgumentEnum.DYM_DIMS]:
        batch_index = get_batch_index(npu_dump_data_path)
        for root, _, files in os.walk(golden_dump_data_path):
            for file_name in files:
                first_dot_index = file_name.find(DOT)
                current_op_name = BATCH_SCENARIO_OP_NAME.format(file_name[:first_dot_index], batch_index)
                dst_file_name = current_op_name + file_name[first_dot_index:]
                shutil.copy(os.path.join(root, file_name), os.path.join(root, dst_file_name))


def get_batch_index_from_name(name):
    batch_index = ""
    last_batch_field_index = name.rfind(ASCEND_BATCH_FIELD)
    pos = last_batch_field_index + len(ASCEND_BATCH_FIELD)
    while pos < len(name) and name[pos].isdigit():
        batch_index += name[pos]
        pos += 1
    return batch_index


def get_data_len_by_shape(shape):
    data_len = 1
    for item in shape:
        if item is -1:
            print_error_log("please check your input shape, one dim in shape is -1.")
            return -1
        data_len = data_len * item
    return data_len


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')
