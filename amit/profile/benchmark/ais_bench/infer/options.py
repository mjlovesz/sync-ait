from multiprocessing import Pool
from multiprocessing import Manager
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
        return check_positive_integer(value)


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


opt_model = click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    ),
    help="the path of the om model"
)


opt_input = click.option(
    '-i',
    '--input',
    'input',
    default=None,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    ),
    help='input file or dir'
)


opt_output = click.option(
    '-o',
    '--output',
    'output',
    default=None,
    type=click.Path(
        path_type=pathlib.Path
    ),
    help='Inference data output path. The inference results are output to '
        'the subdirectory named current date under given output path'
)


opt_output_dirname = click.option(
    '--output_dirname',
    'output_dirname',
    type=str,
    help='actual output directory name. Used with parameter output, cannot be used alone. '
        'The inference result is output to  subdirectory named by output_dirname under  output path. '
        'such as --output_dirname "tmp", the final inference results are output to the folder of  {$output}/tmp'
)


opt_outfmt = click.option(
    '--outfmt',
    default='BIN',
    type=click.Choice(['NPY', 'BIN', 'TXT']),
    help='Output file format (NPY or BIN or TXT)'
)


opt_loop = click.option(
    '-l',
    '--loop',
    default=1,
    type=int,
    callback=check_positive_integer,
    help='the round of the PureInfer.'
)


opt_debug = click.option(
    '--debug',
    default=False,
    type=str,
    callback=str2bool,
    help='Debug switch,print model information'
)


opt_device = click.option(
    '-d',
    '--device',
    default=0,
    type=str,
    callback=check_device_range_valid,
    help='the NPU device ID to use.valid value range is [0, 255]'
)


opt_dymBatch = click.option(
    '--dymBatch',
    'dymBatch',
    default=0,
    type=int,
    help='dynamic batch size param, such as --dymBatch 2'
)


opt_dymHW = click.option(
    '--dymHW',
    'dymHW',
    default=None,
    type=str,
    help='dynamic image size param, such as --dymHW \"300,500\"'
)


opt_dymDims = click.option(
    '--dymDims',
    'dymDims',
    default=None,
    type=str,
    help='dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\"'
)

opt_dymShape = click.option(
    '--dymShape',
    'dymShape',
    type=str,
    default=None,
    help='dynamic shape param, such as --dymShape \"data:1,600;img_info:1,600\"'
)

opt_outputSize = click.option(
    '--outputSize',
    'outputSize',
    default=None,
    type=str,
    help='output size for dynamic shape mode'
)


opt_auto_set_dymshape_mode = click.option(
    '--auto_set_dymshape_mode',
    default=False,
    callback=str2bool,
    help='auto_set_dymshape_mode'
)


opt_auto_set_dymdims_mode = click.option(
    '--auto_set_dymdims_mode',
    default=False,
    callback=str2bool,
    help='auto_set_dymdims_mode'
)


opt_batchsize = click.option(
    '--batchsize',
    default=None,
    callback=check_batchsize_valid,
    help='batch size of input tensor'
)


opt_pure_data_type = click.option(
    '--pure_data_type',
    default="zero",
    type=click.Choice(["zero", "random"]),
    help='null data type for pure inference(zero or random)'
)


opt_profiler = click.option(
    '--profiler',
    default=False,
    type=str,
    callback=str2bool,
    help='profiler switch'
)


opt_dump = click.option(
    '--dump',
    default=False,
    type=str,
    callback=str2bool,
    help='dump switch'
)


opt_acl_json_path = click.option(
    '--acl_json_path',
    default=None,
    type=str,
    help='acl json path for profiling or dump'
)


opt_output_batchsize_axis = click.option(
    '--output_batchsize_axis',
    default=0,
    type=int,
    callback=check_nonnegative_integer,
    help='splitting axis number when outputing tensor results, such as --output_batchsize_axis 1'
)


opt_run_mode = click.option(
    '--run_mode',
    default="array",
    type=click.Choice(["array", "files", "tensor", "full"]),
    help='run mode'
)


opt_display_all_summary = click.option(
    '--display_all_summary',
    default=False,
    type=str,
    callback=str2bool,
    help='display all summary include h2d d2h info'
)


opt_warmup_count = click.option(
    '--warmup_count',
    default=1,
    type=int,
    callback=check_nonnegative_integer,
    help='warmup count before inference'
)


opt_dymShape_range = click.option(
    '--dymShape_range',
    'dymShape_range',
    default=None,
    type=str,
    help='dynamic shape range, such as --dymShape_range "data:1,600~700;img_info:1,600-700"'
)