import os
import re
import argparse
from ais_bench.infer.path_security_check import FileStat

OM_MODEL_MAX_SIZE = 10 * 1024 * 1024 * 1024 # 10GB
ACL_JSON_MAX_SIZE = 8 * 1024 # 8KB
AIPP_CONFIG_MAX_SIZE = 12.5 * 1024 # 12.5KB


def check_dym_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.-]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym string \"{dym_string}\" is not a legal string")
    return dym_string


def check_dym_range_string(value):
    if not value:
        return value
    dym_string = value
    regex = re.compile(r"[^_A-Za-z0-9,;:/.\-~]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string


def check_number_list(value):
    if not value:
        return value
    number_list = value
    regex = re.compile(r"[^0-9,;]")
    if regex.search(number_list):
        raise argparse.ArgumentTypeError(f"number_list \"{number_list}\" is not a legal list")
    return number_list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')


def check_positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_batchsize_valid(value):
    # default value is None
    if value is None:
        return value
    # input value no None
    else:
        return check_positive_integer(value)


def check_nonnegative_integer(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative int value" % value)
    return ivalue


def check_npu_id_range_vaild(value):
    # if contain , split to int list
    min_value = 0
    max_value = 2048
    if ',' in value:
        ilist = [int(v) for v in value.split(',')]
        for ivalue in ilist:
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("{} of npu_id:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, value, min_value, max_value))
        return ilist
    else:
        # default as single int value
        ivalue = int(value)
        if ivalue < min_value or ivalue > max_value:
            raise argparse.ArgumentTypeError("npu_id:{} is invalid. valid value range is [{}, {}]".format(
                ivalue, min_value, max_value))
        return ivalue


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    try:
        # Check if the value contains a comma; if so, split into a list of integers
        if ',' in value:
            ilist = [int(v) for v in value.split(',')]
            for ivalue in ilist:
                if ivalue < min_value or ivalue > max_value:
                    raise argparse.ArgumentTypeError("{} of device:{} is invalid. valid value range is [{}, {}]".format(
                        ivalue, value, min_value, max_value))
            return ilist
        else:
            # default as single int value
            ivalue = int(value)
            if ivalue < min_value or ivalue > max_value:
                raise argparse.ArgumentTypeError("device:{} is invalid. valid value range is [{}, {}]".format(
                    ivalue, min_value, max_value))
            return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError("Argument npu-id invalid input value: {}. "
                                         "Please provide a valid integer or a comma-separated list of integers.".format(value))



def check_om_path_legality(value):
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["om"]):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(OM_MODEL_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read'):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return value


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"weight path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write"):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_acl_json_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["json"]):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(ACL_JSON_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    return path_value


def check_aipp_config_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal('read'):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_type(["config"]):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    if not file_stat.is_legal_file_size(AIPP_CONFIG_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    return path_value