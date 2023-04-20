import logging
import os
import sys
import time
import shutil
import copy
from multiprocessing import Pool
from multiprocessing import Manager
import pathlib

import click
from tqdm import tqdm
from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.io_oprations import (create_infileslist_from_inputs_list,
                                    create_intensors_from_infileslist,
                                    get_narray_from_files_list,
                                    get_tensor_from_files_list,
                                    convert_real_files,
                                    pure_infer_fake_file, save_tensors_to_file)
from ais_bench.infer.summary import summary
from ais_bench.infer.utils import logger
from ais_bench.infer.miscellaneous import dymshape_range_run, get_acl_json_path, version_check, get_batchsize

def set_session_options(session, args):
    # 增加校验
    if args.dymBatch != 0:
        session.set_dynamic_batchsize(args.dymBatch)
    elif args.dymHW !=None:
        hwstr = args.dymHW.split(",")
        session.set_dynamic_hw((int)(hwstr[0]), (int)(hwstr[1]))
    elif args.dymDims !=None:
        session.set_dynamic_dims(args.dymDims)
    elif args.dymShape !=None:
        session.set_dynamic_shape(args.dymShape)
    else:
        session.set_staticbatch()

    if args.batchsize == None:
        args.batchsize = get_batchsize(session, args)
        logger.info("try get model batchsize:{}".format(args.batchsize))

    # 设置custom out tensors size
    if args.outputSize != None:
        customsizes = [int(n) for n in args.outputSize.split(',')]
        logger.debug("set customsize:{}".format(customsizes))
        session.set_custom_outsize(customsizes)

def init_inference_session(args):
    acl_json_path = get_acl_json_path(args)
    session = InferSession(args.device, args.model, acl_json_path, args.debug, args.loop)

    set_session_options(session, args)
    logger.debug("session info:{}".format(session.session))
    return session

def set_dymshape_shape(session, inputs):
    l = []
    intensors_desc = session.get_inputs()
    for i, input in enumerate(inputs):
        str_shape = [ str(shape) for shape in input.shape ]
        dyshape = "{}:{}".format(intensors_desc[i].name, ",".join(str_shape))
        l.append(dyshape)
    dyshapes = ';'.join(l)
    logger.debug("set dymshape shape:{}".format(dyshapes))
    session.set_dynamic_shape(dyshapes)
    summary.add_batchsize(inputs[0].shape[0])

def set_dymdims_shape(session, inputs):
    l = []
    intensors_desc = session.get_inputs()
    for i, input in enumerate(inputs):
        str_shape = [ str(shape) for shape in input.shape ]
        dydim = "{}:{}".format(intensors_desc[i].name, ",".join(str_shape))
        l.append(dydim)
    dydims = ';'.join(l)
    logger.debug("set dymdims shape:{}".format(dydims))
    session.set_dynamic_dims(dydims)
    summary.add_batchsize(inputs[0].shape[0])

def warmup(session, args, intensors_desc, infiles):
    # prepare input data
    infeeds = []
    for j, files in enumerate(infiles):
        if args.run_mode == "tensor":
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize, args.pure_data_type, args.no_combine_tensor_mode)
            infeeds.append(tensor)
        else:
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize, args.pure_data_type, args.no_combine_tensor_mode)
            infeeds.append(narray)
    session.set_loop_count(1)
    # warmup
    for i in range(args.warmup_count):
        outputs = run_inference(session, args, infeeds, out_array=True)

    session.set_loop_count(args.loop)

    # reset summary info
    summary.reset()
    session.reset_sumaryinfo()
    MemorySummary.reset()
    logger.info("warm up {} done".format(args.warmup_count))

def run_inference(session, args, inputs, out_array=False):
    if args.auto_set_dymshape_mode == True:
        set_dymshape_shape(session, inputs)
    elif args.auto_set_dymdims_mode == True:
        set_dymdims_shape(session, inputs)
    outputs = session.run(inputs, out_array)
    return outputs

# tensor to loop infer
def infer_loop_tensor_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference tensor Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize, args.pure_data_type, args.no_combine_tensor_mode)
            intensors.append(tensor)
        outputs = run_inference(session, args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix != None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i, args.output_batchsize_axis)

# files to loop iner
def infer_loop_files_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference files Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            real_files = convert_real_files(files)
            tensor = session.create_tensor_from_fileslist(intensors_desc[j], real_files)
            intensors.append(tensor)
        outputs = run_inference(session, args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix != None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i, args.output_batchsize_axis)

# First prepare the data, then execute the reference, and then write the file uniformly
def infer_fulltensors_run(session, args, intensors_desc, infileslist, output_prefix):
    outtensors = []
    intensorslist = create_intensors_from_infileslist(infileslist, intensors_desc, session, args.pure_data_type, args.no_combine_tensor_mode)

    #for inputs in intensorslist:
    for inputs in tqdm(intensorslist, file=sys.stdout, desc='Inference Processing full'):
        outputs = run_inference(session, args, inputs)
        outtensors.append(outputs)

    for i, outputs in enumerate(outtensors):
        session.convert_tensors_to_host(outputs)
        if output_prefix != None:
            save_tensors_to_file(outputs, output_prefix, infileslist[i], args.outfmt, i, args.output_batchsize_axis)

# loop numpy array to infer
def infer_loop_array_run(session, args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference array Processing')):
        innarrays = []
        for j, files in enumerate(infiles):
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize, args.pure_data_type)
            innarrays.append(narray)
        outputs = run_inference(session, args, innarrays)
        session.convert_tensors_to_host(outputs)
        if args.output != None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i, args.output_batchsize_axis)

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


def check_positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


arg_model = click.argument(
    'model',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    )
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
    help='Inference data output path. The inference results are output to the subdirectory named current date under given output path'
)


opt_output_dirname = click.option(
    '--output_dirname',
    'output_dirname',
    type=str,
    help='actual output directory name. Used with parameter output, cannot be used alone. The inference result is output to  subdirectory named by output_dirname under  output path. such as --output_dirname "tmp", the final inference results are output to the folder of  {$output}/tmp'
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
    type=int,
    callback=check_device_range_valid,
    help='the NPU device ID to use.valid value range is [0, 255]'
)


opt_dymBatch = click.option(
    '--dymBatch',
    default=0,
    type=int,
    help='dynamic batch size param, such as --dymBatch 2'
)


opt_dymHW = click.option(
    '--dymHW',
    default=None,
    type=str,
    help='dynamic image size param, such as --dymHW \"300,500\"'
)


opt_dymDims = click.option(
    '--dymDims',
    default=None,
    type=str,
    help='dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\"'
)


opt_outputSize = click.option(
    '--outputSize',
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
    type=click.Choice("zero", "random"),
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
    default=None,
    type=str,
    help='dynamic shape range, such as --dymShape_range "data:1,600~700;img_info:1,600-700"'
)
