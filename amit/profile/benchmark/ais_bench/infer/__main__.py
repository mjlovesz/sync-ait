import argparse

from profile.benchmark.ais_bench.infer.utils import logger
from profile.benchmark.ais_bench.infer.main_enter import main_enter
from profile.benchmark.ais_bench.infer.args_adapter import MyArgs
from profile.benchmark.options import (
    str2bool,
    check_positive_integer,
    check_device_range_valid,
    check_batchsize_valid,
    check_nonnegative_integer
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="the path of the om model")
    parser.add_argument("--input", "-i", default=None, help="input file or dir")
    parser.add_argument("--output", "-o", default=None, help="Inference data output path. The inference results are output to the subdirectory named current date under given output path")
    parser.add_argument("--output_dirname", type=str, default=None, help="actual output directory name. Used with parameter output, cannot be used alone. The inference result is output to  subdirectory named by output_dirname under  output path. such as --output_dirname 'tmp', the final inference results are output to the folder of  {$output}/tmp")
    parser.add_argument("--outfmt", default="BIN", choices=["NPY", "BIN", "TXT"], help="Output file format (NPY or BIN or TXT)")
    parser.add_argument("--loop", "-l", type=check_positive_integer, default=1, help="the round of the PureInfer.")
    parser.add_argument("--debug", type=str2bool, default=False, help="Debug switch,print model information")
    parser.add_argument("--device", "-d", type=check_device_range_valid, default=0, help="the NPU device ID to use.valid value range is [0, 255]")
    parser.add_argument("--dymBatch", type=int, default=0, help="dynamic batch size param，such as --dymBatch 2")
    parser.add_argument("--dymHW", type=str, default=None, help="dynamic image size param, such as --dymHW \"300,500\"")
    parser.add_argument("--dymDims", type=str, default=None, help="dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\"")
    parser.add_argument("--dymShape", type=str, default=None, help="dynamic shape param, such as --dymShape \"data:1,600;img_info:1,600\"")
    parser.add_argument("--outputSize", type=str, default=None, help="output size for dynamic shape mode")
    parser.add_argument("--auto_set_dymshape_mode", type=str2bool, default=False, help="auto_set_dymshape_mode")
    parser.add_argument("--auto_set_dymdims_mode", type=str2bool, default=False, help="auto_set_dymdims_mode")
    parser.add_argument("--batchsize", type=check_batchsize_valid, default=None, help="batch size of input tensor")
    parser.add_argument("--pure_data_type", type=str, default="zero", choices=["zero", "random"], help="null data type for pure inference(zero or random)")
    parser.add_argument("--profiler", type=str2bool, default=False, help="profiler switch")
    parser.add_argument("--dump", type=str2bool, default=False, help="dump switch")
    parser.add_argument("--acl_json_path", type=str, default=None, help="acl json path for profiling or dump")
    parser.add_argument("--output_batchsize_axis",  type=check_nonnegative_integer, default=0, help="splitting axis number when outputing tensor results, such as --output_batchsize_axis 1")
    parser.add_argument("--run_mode", type=str, default="array", choices=["array", "files", "tensor", "full"], help="run mode")
    parser.add_argument("--display_all_summary", type=str2bool, default=False, help="display all summary include h2d d2h info")
    parser.add_argument("--warmup_count",  type=check_nonnegative_integer, default=1, help="warmup count before inference")
    parser.add_argument("--dymShape_range", type=str, default=None, help="dynamic shape range, such as --dymShape_range \"data:1,600~700;img_info:1,600-700\"")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    my_args = MyArgs(args.model, args.input, args.output, args.output_dirname, args.outfmt, args.loop, args.debug, args.device, 
                args.dymBatch, args.dymHW, args.dymDims, args.dymShape, args.outputSize, args.auto_set_dymshape_mode, 
                args.auto_set_dymdims_mode, args.batchsize, args.pure_data_type, args.profiler, args.dump, 
                args.acl_json_path, args.output_batchsize_axis, args.run_mode, args.display_all_summary, 
                args.warmup_count, args.dymShape_range)

    main_enter(my_args)
