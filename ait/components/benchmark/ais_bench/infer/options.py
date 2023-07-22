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

import pathlib
import argparse

import click


def str2bool(ctx, param, v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected true, 1, false, 0 with case insensitive.')


def check_positive_integer(ctx, param, value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_batchsize_valid(ctx, param, value):
    # default value is None
    if value is None:
        return value
    # input value no None
    else:
        return check_positive_integer(ctx, param, value)


def check_nonnegative_integer(ctx, param, value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonnegative int value" % value)
    return ivalue


def check_device_range_valid(ctx, param, value):
    # if contain , split to int list
    min_value = 0
    max_value = 255
    if ',' in value:
        ilist = [ int(v) for v in value.split(',') ]
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


def check_args(ctx: click.Context, params: click.Option, value: str):
    """
    check whether the param is provided
    """
    args = [
        opt
        for param in ctx.command.params
        for opt in param.opts
    ]
    if value in args:
        raise click.MissingParameter()
    return value


opt_model = click.option(
    "-om",
    "--om-model",
    "om_model",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    ),
    callback=check_args,
    help="The path of the om model"
)


opt_input_path = click.option(
    '-i',
    '--input',
    'input_path',
    default=None,
    callback=check_args,
    help='Input file or dir'
)


opt_output = click.option(
    '-o',
    '--output',
    'output',
    default=None,
    type=click.Path(
        path_type=pathlib.Path
    ),
    callback=check_args,
    help='Inference data output path. The inference results are output to '
        'the subdirectory named current date under given output path'
)


opt_output_dirname = click.option(
    '-od',
    '--output-dirname',
    'output_dirname',
    type=str,
    callback=check_args,
    help='Actual output directory name. Used with parameter output, cannot be used alone. '
        'The inference result is output to subdirectory named by output_dirname under output path. '
        'such as --output_dirname "tmp", the final inference results are output to the folder of {$output}/tmp'
)


opt_outfmt = click.option(
    '--outfmt',
    default='BIN',
    type=click.Choice(['NPY', 'BIN', 'TXT']),
    help='Output file format (NPY or BIN or TXT)'
)


opt_loop = click.option(
    '--loop',
    default=1,
    type=int,
    callback=check_positive_integer,
    help='The round of the PureInfer.'
)


opt_debug = click.option(
    '--debug',
    default=False,
    type=str,
    callback=str2bool,
    help='Debug switch, print model information'
)


opt_device = click.option(
    '-d',
    '--device',
    default=0,
    type=str,
    callback=check_device_range_valid,
    help='The NPU device ID to use.valid value range is [0, 255]'
)


opt_dym_batch = click.option(
    '-db',
    '--dym-batch',
    'dym_batch',
    default=0,
    type=int,
    help='Dynamic batch size param, such as --dym_batch 2'
)


opt_dym_hw = click.option(
    '-dhw',
    '--dym-hw',
    'dym_hw',
    default=None,
    type=str,
    callback=check_args,
    help='Dynamic image size param, such as --dym_hw \"300,500\"'
)


opt_dym_dims = click.option(
    '-dd',
    '--dym-dims',
    'dym_dims',
    default=None,
    type=str,
    callback=check_args,
    help='Dynamic dims param, such as --dym_dims \"data:1,600;img_info:1,600\"'
)

opt_dym_shape = click.option(
    '-ds',
    '--dym-shape',
    'dym_shape',
    type=str,
    default=None,
    callback=check_args,
    help='Dynamic shape param, such as --dym_shape \"data:1,600;img_info:1,600\"'
)


opt_dym_shape_range = click.option(
    '-dr',
    '--dym-shape-range',
    'dym_shape_range',
    default=None,
    type=str,
    callback=check_args,
    help='Dynamic shape range, such as --dym_shape_range "data:1,600~700;img_info:1,600-700"'
)

opt_output_size = click.option(
    '-outsize',
    '--output-size',
    'output_size',
    default=None,
    type=str,
    callback=check_args,
    help='Output size for dynamic shape mode'
)


opt_auto_set_dymshape_mode = click.option(
    '-asdsm',
    '--auto-set-dymshape-mode',
    'auto_set_dymshape_mode',
    default=False,
    callback=str2bool,
    help='Auto_set_dymshape_mode'
)


opt_auto_set_dymdims_mode = click.option(
    '-asddm',
    '--auto-set-dymdims-mode',
    'auto_set_dymdims_mode',
    default=False,
    callback=str2bool,
    help='Auto_set_dymdims_mode'
)


opt_batchsize = click.option(
    '--batch-size',
    'batch_size',
    default=None,
    callback=check_batchsize_valid,
    help='Batch size of input tensor'
)


opt_pure_data_type = click.option(
    '-pdt',
    '--pure-data-type',
    'pure_data_type',
    default="zero",
    type=click.Choice(["zero", "random"]),
    help='Null data type for pure inference(zero or random)'
)


opt_profiler = click.option(
    '-pf',
    '--profiler',
    default=False,
    type=str,
    callback=str2bool,
    help='Profiler switch'
)

opt_profiler_rename = click.option(
    '--profiler_rename',
    default=True,
    type=str,
    callback=str2bool,
    help='Profiler rename switch'
)

opt_dump = click.option(
    '--dump',
    default=False,
    type=str,
    callback=str2bool,
    help='Dump switch'
)


opt_acl_json_path = click.option(
    '-acl',
    '--acl-json-path',
    'acl_json_path',
    default=None,
    type=str,
    callback=check_args,
    help='Acl json path for profiling or dump'
)


opt_output_batchsize_axis = click.option(
    '-oba',
    '--output-batchsize-axis',
    'output_batchsize_axis',
    default=0,
    type=int,
    callback=check_nonnegative_integer,
    help='Splitting axis number when outputing tensor results, such as --output_batchsize_axis 1'
)


opt_run_mode = click.option(
    '-rm',
    '--run-mode',
    'run_mode',
    default="array",
    type=click.Choice(["array", "files", "tensor", "full"]),
    help='Run mode'
)


opt_display_all_summary = click.option(
    '-das',
    '--display-all-summary',
    'display_all_summary',
    default=False,
    type=str,
    callback=str2bool,
    help='Display all summary include h2d d2h info'
)


opt_warmup_count = click.option(
    '-wcount',
    '--warmup-count',
    'warmup_count',
    default=1,
    type=int,
    callback=check_nonnegative_integer,
    help='Warmup count before inference'
)


opt_aipp_config = click.option(
    '-aipp',
    '--aipp-config',
    'aipp_config',
    type=str,
    default=None,
    callback=check_args,
    help="File type: .config, to set actual aipp params before infer"
)

opt_energy_consumption = click.option(
    '-ec',
    '--energy_consumption',
    'energy_consumption',
    default=False,
    type=str,
    callback=str2bool,
    help="Obtain power consumption data for model inference"
)

opt_npu_id = click.option(
    '--npu_id',
    'npu_id',
    default=0,
    type=str,
    callback=check_device_range_valid,
    help="The NPU ID to use.valid value range is [0, 255]"
)

opt_backend = click.option(
    "--backend",
    type=str,
    default=None,
    callback=check_args,
    help="Backend trtexec"
)

opt_perf = click.option(
    "--perf",
    type=str,
    callback=str2bool,
    default=False,
    help="Perf switch"
)

opt_pipeline = click.option(
    '--pipeline',
    default=False,
    type=str,
    callback=str2bool,
    help='Pipeline switch'
)