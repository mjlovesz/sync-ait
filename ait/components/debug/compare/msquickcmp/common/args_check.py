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
import re
import argparse
from ait.components.utils.file_open_check import FileStat, args_path_output_check

STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")
MAX_SIZE_LIMITE_NORMAL_MODEL = 10 * 1024 * 1024 * 1024 # 10GB
MAX_SIZE_LIMITE_FUSION_FILE = 1 * 1024 * 1024 * 1024 # 1GB

def check_model_path_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check(["onnx", "prototxt", "pb"]):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"model path:{path_value} is illegal. Please check.")
    return path_value


def check_om_path_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check(["om"]):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    return path_value


def check_weight_path_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check(["caffemodel"]):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = str(value).split(',')
    for input_path in inputs_list:
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal([os.R_OK]):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return str(value)


def check_directory_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"cann path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"cann path:{path_value} is illegal. Please check.")
    if not file_stat.is_dir:
        raise argparse.ArgumentTypeError(f"om path:{path_value} is not a directory. Please check.")
    return path_value


def check_output_path_legality(value):
    if not value:
        return value
    path_value = str(value)
    if not args_path_output_check(path_value):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_dict_kind_string(value):
    # just like "input_name1:1,224,224,3;input_name2:3,300"
    if not value:
        return value
    input_shape = str(value)
    regex = re.compile(r"[^_A-Za-z0-9,;:]")
    if regex.search(input_shape):
        raise argparse.ArgumentTypeError(f"dym string \"{input_shape}\" is not a legal string")
    return input_shape


def check_device_range_valid(value):
    min_value = 0
    max_value = 255
    ivalue = int(value)
    if ivalue < min_value or ivalue > max_value:
        raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
            ivalue, min_value, max_value))
    return ivalue


def check_number_list(value):
    # just like "1241414,124141,124424"
    if not value:
        return value
    outsize_list = str(value).split(',')
    for outsize in outsize_list:
        regex = re.compile(r"[^0-9]")
        if regex.search(outsize):
            raise argparse.ArgumentTypeError(f"output size \"{outsize}\" is not a legal string")
    return str(value)


def check_dym_range_string(value):
    if not value:
        return value
    dym_string = str(value)
    regex = re.compile(r"[^_A-Za-z0-9\-~,;:]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string


def check_fusion_cfg_path_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check("cfg"):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"fusion switch file path:{path_value} is illegal. Please check.")
    return path_value


def check_quant_json_path_legality(value):
    path_value = str(value)
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check("json"):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(MAX_SIZE_LIMITE_NORMAL_MODEL):
        raise argparse.ArgumentTypeError(f"quant file path:{path_value} is illegal. Please check.")
    return path_value


def safe_string(value):
    if re.search(STR_WHITE_LIST_REGEX, value):
        raise ValueError("String parameter contains invalid characters.")
    return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')
