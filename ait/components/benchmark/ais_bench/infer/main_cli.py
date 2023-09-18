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

from components.utils.parser import BaseCommand
from ais_bench.infer.benchmark_process import benchmark_process
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.path_security_check import args_path_output_check, InFileStat

OM_MODEL_MAX_SIZE = 10 * 1024 * 1024 * 1024 # 10GB
ACL_JSON_MAX_SIZE = 8 * 1024 # 8KB
AIPP_CONFIG_MAX_SIZE = 12.5 * 1024 # 12.5KB


def dym_string_check(value):
    if not value:
        return None
    dym_string = str(value)
    regex = re.compile(r"[^_A-Za-z0-9,;:]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym string \"{dym_string}\" is not a legal string")
    return dym_string


def dym_range_string_check(value):
    if not value:
        return None
    dym_string = str(value)
    regex = re.compile(r"[^_A-Za-z0-9\-~,;:]")
    if regex.search(dym_string):
        raise argparse.ArgumentTypeError(f"dym range string \"{dym_string}\" is not a legal string")
    return dym_string


def number_list_check(value):
    if not value:
        return None
    number_list = str(value)
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


def check_device_range_valid(value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
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


def check_om_path_legality(value):
    path_value = str(value)
    try:
        file_stat = InFileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check("om"):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(OM_MODEL_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"om path:{path_value} is illegal. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return None
    inputs_list = str(value).split(',')
    for input_path in inputs_list:
        try:
            file_stat = InFileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"acl json path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal([os.R_OK]):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return str(value)


def check_output_path_legality(value):
    if not value:
        return None
    path_value = str(value)
    if not args_path_output_check(path_value, [os.R_OK]):
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.")
    return path_value


def check_acl_json_path_legality(value):
    if not value:
        return None
    path_value = str(value)
    try:
        file_stat = InFileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_type_check("json"):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(ACL_JSON_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"acl json path:{path_value} is illegal. Please check.")
    return path_value


def check_aipp_config_path_legality(value):
    if not value:
        return None
    path_value = str(value)
    try:
        file_stat = InFileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal([os.R_OK]):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    if not not file_stat.path_file_type_check("config"):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    if not file_stat.path_file_size_check(AIPP_CONFIG_MAX_SIZE):
        raise argparse.ArgumentTypeError(f"aipp config path:{path_value} is illegal. Please check.")
    return path_value


class BenchmarkCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "-om",
            "--om-model",
            type=check_om_path_legality,
            required=True,
            help="The path of the om model"
        )
        parser.add_argument(
            '-i',
            '--input',
            type=check_input_path_legality,
            default=None,
            help="Input file or dir"
        )
        parser.add_argument(
            '-o',
            '--output',
            type=check_output_path_legality,
            default=None,
            help="Inference data output path. The inference results are output to \
                the subdirectory named current date under given output path"
        )
        parser.add_argument(
            '-od',
            "--output-dirname",
            type=check_output_path_legality,
            default=None,
            help="Actual output directory name. \
                Used with parameter output, cannot be used alone. \
                The inference result is output to subdirectory named by output_dirname \
                under  output path. such as --output_dirname 'tmp', \
                the final inference results are output to the folder of  {$output}/tmp"
        )
        parser.add_argument(
            "--outfmt",
            default="BIN",
            choices=["NPY", "BIN", "TXT"],
            help="Output file format (NPY or BIN or TXT)"
        )
        parser.add_argument(
            "--loop",
            type=check_positive_integer,
            default=1,
            help="The round of the PureInfer."
        )
        parser.add_argument(
            "--debug",
            type=str2bool,
            default=False,
            help="Debug switch,print model information"
        )
        parser.add_argument(
            "-d",
            "--device",
            type=check_device_range_valid,
            default=0,
            help="The NPU device ID to use.valid value range is [0, 255]"
        )
        parser.add_argument(
            '-db',
            '--dym-batch',
            dest="dym_batch",
            type=check_positive_integer,
            default=0,
            help="Dynamic batch size param，such as --dymBatch 2"
        )
        parser.add_argument(
            '-dhw',
            '--dym-hw',
            dest="dym_hw",
            type=dym_string_check,
            default=None,
            help="Dynamic image size param, such as --dymHW \"300,500\""
        )
        parser.add_argument(
            '-dd',
            '--dym-dims',
            dest="dym_dims",
            type=dym_string_check,
            default=None,
            help="Dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\""
        )
        parser.add_argument(
            '-ds',
            '--dym-shape',
            dest="dym_shape",
            type=dym_string_check,
            default=None,
            help="Dynamic shape param, such as --dymShape \"data:1,600;img_info:1,600\""
        )
        parser.add_argument(
            '-outsize',
            '--output-size',
            dest="output_size",
            type=number_list_check,
            default=None,
            help="Output size for dynamic shape mode"
        )
        parser.add_argument(
            '-asdsm',
            '--auto-set-dymshape-mode',
            dest='auto_set_dymshape_mode',
            type=str2bool,
            default=False,
            help="Auto_set_dymshape_mode"
        )
        parser.add_argument(
            '-asddm',
            '--auto-set-dymdims-mode',
            dest='auto_set_dymdims_mode',
            type=str2bool,
            default=False,
            help="Auto_set_dymdims_mode"
        )
        parser.add_argument(
            '--batch-size',
            type=check_batchsize_valid,
            default=None,
            help="Batch size of input tensor"
        )
        parser.add_argument(
            '-pdt',
            '--pure-data-type',
            dest='pure_data_type',
            type=str,
            default="zero",
            choices=["zero", "random"],
            help="Null data type for pure inference(zero or random)"
        )
        parser.add_argument(
            '-pf',
            '--profiler',
            type=str2bool,
            default=False,
            help="Profiler switch"
        )
        parser.add_argument(
            "--dump",
            type=str2bool,
            default=False,
            help="Dump switch"
        )
        parser.add_argument(
            '-acl',
            '--acl-json-path',
            dest='acl_json_path',
            type=check_acl_json_path_legality,
            default=None,
            help="Acl json path for profiling or dump"
        )
        parser.add_argument(
            '-oba',
            '--output-batchsize-axis',
            dest='output_batchsize_axis',
            type=check_nonnegative_integer,
            default=0,
            help="Splitting axis number when outputing tensor results, such as --output_batchsize_axis 1"
        )
        parser.add_argument(
            '-rm',
            '--run-mode',
            dest='run_mode',
            type=str,
            default="array",
            choices=["array", "files", "tensor", "full"],
            help="Run mode"
        )
        parser.add_argument(
            '-das',
            '--display-all-summary',
            dest='display_all_summary',
            type=str2bool,
            default=False,
            help="Display all summary include h2d d2h info"
        )
        parser.add_argument(
            '-wcount',
            '--warmup-count',
            dest='warmup_count',
            type=check_nonnegative_integer,
            default=1,
            help="Warmup count before inference"
        )
        parser.add_argument(
            '-dr',
            '--dym-shape-range',
            dest="dym_shape_range",
            type=dym_range_string_check,
            default=None,
            help='Dynamic shape range, such as --dym_shape_range "data:1,600~700;img_info:1,600-700"'
        )
        parser.add_argument(
            '-aipp',
            '--aipp-config',
            dest='aipp_config',
            type=check_aipp_config_path_legality,
            default=None,
            help="File type: .config, to set actual aipp params before infer"
        )
        parser.add_argument(
            '-ec',
            '--energy-consumption',
            dest='energy_consumption',
            type=str2bool,
            default=False,
            help="Obtain power consumption data for model inference"
        )
        parser.add_argument(
            '--npu-id',
            dest='npu_id',
            type=check_nonnegative_integer,
            default=0,
            help="The NPU ID to use. using cmd: \'npu-smi info\' to check "
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=None,
            choices=["trtexec"],
            help="Backend trtexec"
        )
        parser.add_argument(
            "--perf",
            type=str2bool,
            default=False,
            help="Perf switch"
        )
        parser.add_argument(
            "--pipeline",
            type=str2bool,
            default=False,
            help="Pipeline switch"
        )
        parser.add_argument(
            "--profiler-rename",
            type=str2bool,
            default=True,
            help="Profiler rename switch"
        )
        parser.add_argument(
            "--dump-npy",
            type=str2bool,
            default=False,
            help="dump data convert to npy"
        )
        parser.add_argument(
            '--divide-input',
            dest='divide_input',
            type=str2bool,
            default=False,
            help='Input datas need to be divided to match multi devices or not, \
                --device should be list, default False'
        )
        parser.add_argument(
            '--thread',
            dest='thread',
            type=check_positive_integer,
            default=1,
            help="Number of thread for computing. \
                need to set --pipeline when setting thread number to be more than one."
        )

    def handle(self, args):
        args = BenchMarkArgsAdapter(args.om_model, args.input, args.output, args.output_dirname, args.outfmt,
                                    args.loop, args.debug, args.device, args.dym_batch, args.dym_hw, args.dym_dims,
                                    args.dym_shape, args.output_size, args.auto_set_dymshape_mode,
                                    args.auto_set_dymdims_mode, args.batch_size, args.pure_data_type, args.profiler,
                                    args.dump, args.acl_json_path, args.output_batchsize_axis, args.run_mode,
                                    args.display_all_summary, args.warmup_count, args.dym_shape_range,
                                    args.aipp_config, args.energy_consumption, args.npu_id, args.backend, args.perf,
                                    args.pipeline, args.profiler_rename, args.dump_npy, args.divide_input, args.thread)
        benchmark_process(args)


def get_cmd_instance():
    help_info = "benchmark tool to get performance data including latency and throughput"
    cmd_instance = BenchmarkCommand("benchmark", help_info)
    return cmd_instance