import argparse

from ais_bench.infer.utils import logger


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True,
                        help="the path of the om model")
    parser.add_argument("--input", "-i", default=None,
                        help="input file or dir")
    parser.add_argument("--output", "-o", default=None,
                        help="Inference data output path. The inference results are output"
                             " to the subdirectory named current date under given output path")
    parser.add_argument("--output_dirname", type=str, default=None,
                        help="actual output directory name. Used with parameter output, cannot be used alone."
                             "The inference result is output to subdirectory named by output_dirname under output path."
                             " such as --output_dirname 'tmp',"
                             " the final inference results are output to the folder of  {$output}/tmp")
    parser.add_argument("--outfmt", default="BIN", choices=["NPY", "BIN", "TXT"],
                        help="Output file format (NPY or BIN or TXT)")
    parser.add_argument("--loop", "-l", type=check_positive_integer, default=1,
                        help="the round of the PureInfer.")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Debug switch,print model information")
    parser.add_argument("--device", "-d", type=check_device_range_valid, default=0,
                        help="the NPU device ID to use.valid value range is [0, 255]")
    parser.add_argument("--dymBatch", type=int, default=0, dest="dym_batch",
                        help="dynamic batch size param，such as --dymBatch 2")
    parser.add_argument("--dymHW", type=str, default=None, dest="dym_hw",
                        help="dynamic image size param, such as --dymHW \"300,500\"")
    parser.add_argument("--dymDims", type=str, default=None, dest="dym_dims",
                        help="dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\"")
    parser.add_argument("--dymShape", type=str, default=None, dest="dym_shape",
                        help="dynamic shape param, such as --dymShape \"data:1,600;img_info:1,600\"")
    parser.add_argument("--outputSize", type=str, default=None, dest="output_size",
                        help="output size for dynamic shape mode")
    parser.add_argument("--auto_set_dymshape_mode", type=str2bool, default=False,
                        help="auto_set_dymshape_mode")
    parser.add_argument("--auto_set_dymdims_mode", type=str2bool, default=False,
                        help="auto_set_dymdims_mode")
    parser.add_argument("--batchsize", type=check_batchsize_valid, default=None,
                        help="batch size of input tensor")
    parser.add_argument("--pure_data_type", type=str, default="zero", choices=["zero", "random"],
                        help="null data type for pure inference(zero or random)")
    parser.add_argument("--profiler", type=str2bool, default=False,
                        help="profiler switch")
    parser.add_argument("--dump", type=str2bool, default=False,
                        help="dump switch")
    parser.add_argument("--acl_json_path", type=str, default=None,
                        help="acl json path for profiling or dump")
    parser.add_argument("--output_batchsize_axis", type=check_nonnegative_integer, default=0,
                        help="splitting axis number when outputing tensor results, such as --output_batchsize_axis 1")
    parser.add_argument("--run_mode", type=str, default="array", choices=["array", "files", "tensor", "full"],
                        help="run mode")
    parser.add_argument("--display_all_summary", type=str2bool, default=False,
                        help="display all summary include h2d d2h info")
    parser.add_argument("--warmup_count", type=check_nonnegative_integer, default=1,
                        help="warmup count before inference")
    parser.add_argument("--dymShape_range", type=str, default=None, dest="dym_shape_range",
                        help="dynamic shape range, such as --dymShape_range \"data:1,600~700;img_info:1,600-700\"")

    input_args = parser.parse_args()

    if input_args.profiler is True and input_args.dump is True:
        logger.error("parameter --profiler cannot be true at the same time as parameter --dump, please check them!\n")
        raise RuntimeError('error bad parameters --profiler and --dump')

    if (input_args.profiler is True or input_args.dump is True) and (input_args.output is None):
        logger.error("when dump or profiler, miss output path, please check them!")
        raise RuntimeError('miss output parameter!')

    if not input_args.auto_set_dymshape_mode and not input_args.auto_set_dymdims_mode:
        input_args.no_combine_tensor_mode = False
    else:
        input_args.no_combine_tensor_mode = True

    if input_args.profiler is True and input_args.warmup_count != 0 and input_args.input is not None:
        logger.info("profiler mode with input change warmup_count to 0")
        input_args.warmup_count = 0

    if input_args.output is None and input_args.output_dirname is not None:
        logger.error("parameter --output_dirname cann't be used alone."
                     " Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output_dirname')
    return input_args


if __name__ == "__main__":
    args = get_args()

    main_enter(args)