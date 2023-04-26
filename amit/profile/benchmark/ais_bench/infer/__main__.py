import argparse
import logging
import math
import os
import sys

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
from ais_bench.infer.utils import (get_file_content, get_file_datasize,
                            get_fileslist_from_dir, list_split, logger,
                            save_data_to_files)


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

def msprof_run_profiling(args):
    cmd = sys.executable + " " + ' '.join(sys.argv) + " --profiler=0 --warmup_count=0"
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
        msprof_bin, args.output, cmd)
    logger.info("msprof cmd:{} begin run".format(msprof_cmd))
    ret = os.system(msprof_cmd)
    logger.info("msprof cmd:{} end run ret:{}".format(msprof_cmd, ret))

def main(args, index=0, msgq=None, device_list=None):
    # if msgq is not None,as subproces run
    if msgq != None:
        logger.info("subprocess_{} main run".format(index))

    if args.debug == True:
        logger.setLevel(logging.DEBUG)

    session = init_inference_session(args)

    intensors_desc = session.get_inputs()
    if device_list != None and len(device_list) > 1:
        if args.output != None:
            if args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(args.output, timestr)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            else:
                output_prefix = os.path.join(args.output, args.output_dirname)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, 0o755)
            logger.info("output path:{}".format(output_prefix))
        else:
            output_prefix = None
    else:
        if args.output != None:
            if args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(args.output, timestr)
            else:
                output_prefix = os.path.join(args.output, args.output_dirname)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, 0o755)
            logger.info("output path:{}".format(output_prefix))
        else:
            output_prefix = None

    inputs_list = [] if args.input is None else args.input.split(',')

    # create infiles list accord inputs list
    if len(inputs_list) == 0:
        # Pure reference scenario. Create input zero data
        infileslist = [[ [ pure_infer_fake_file ] for index in intensors_desc ]]
    else:
        infileslist = create_infileslist_from_inputs_list(inputs_list, intensors_desc, args.no_combine_tensor_mode)

    warmup(session, args, intensors_desc, infileslist[0])

    if msgq != None:
		# wait subprocess init ready, if time eplapsed,force ready run
        logger.info("subprocess_{} qsize:{} now waiting".format(index, msgq.qsize()))
        msgq.put(index)
        time_sec = 0
        while True:
            if msgq.qsize() >= args.subprocess_count:
                break
            time_sec = time_sec + 1
            if time_sec > 10:
                logger.warning("subprocess_{} qsize:{} time:{} s elapsed".format(index, msgq.qsize(), time_sec))
                break
            time.sleep(1)
        logger.info("subprocess_{} qsize:{} ready to infer run".format(index, msgq.qsize()))

    start_time = time.time()

    if args.run_mode == "array":
        infer_loop_array_run(session, args, intensors_desc, infileslist, output_prefix)
    elif args.run_mode == "files":
        infer_loop_files_run(session, args, intensors_desc, infileslist, output_prefix)
    elif args.run_mode == "full":
        infer_fulltensors_run(session, args, intensors_desc, infileslist, output_prefix)
    elif args.run_mode == "tensor":
        infer_loop_tensor_run(session, args, intensors_desc, infileslist, output_prefix)
    else:
        raise RuntimeError('wrong run_mode:{}'.format(args.run_mode))

    end_time = time.time()

    summary.add_args(sys.argv)
    s = session.sumary()
    summary.npu_compute_time_list = s.exec_time_list
    summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
    summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
    summary.report(args.batchsize, output_prefix, args.display_all_summary)

    if msgq != None:
		# put result to msgq
        msgq.put([index, summary.infodict['throughput'], start_time, end_time])

    session.finalize()

def print_subproces_run_error(value):
    logger.error("subprocess run failed error_callback:{}".format(value))

def seg_input_data_for_multi_process(args, inputs, jobs):
    inputs_list = [] if inputs is None else inputs.split(',')
    if inputs_list == None:
        return inputs_list

    fileslist = []
    if os.path.isfile(inputs_list[0]) == True:
        fileslist = inputs_list
    elif os.path.isdir(inputs_list[0]):
        for dir in inputs_list:
            fileslist.extend(get_fileslist_from_dir(dir))
    else:
        logger.error('error {} not file or dir'.format(inputs_list[0]))
        raise RuntimeError()

    args.device = 0
    session = init_inference_session(args)
    intensors_desc = session.get_inputs()
    chunks_elements = math.ceil(len(fileslist) / len(intensors_desc))
    chunks = list(list_split(fileslist, chunks_elements, None))
    fileslist = [ [] for e in range(jobs) ]
    for i, chunk in enumerate(chunks):
        splits_elements = math.ceil(len(chunk) / jobs)
        splits = list(list_split(chunk, splits_elements, None))
        for j, split in enumerate(splits):
            fileslist[j].extend(split)
    res = []
    for files in fileslist:
        res.append(','.join(list(filter(None, files))))
    return res

def multidevice_run(args):
    logger.info("multidevice:{} run begin".format(args.device))
    device_list = args.device
    p = Pool(len(device_list))
    msgq = Manager().Queue()

    args.subprocess_count = len(device_list)
    jobs = args.subprocess_count
    splits = None
    if (args.input != None):
        splits = seg_input_data_for_multi_process(args, args.input, jobs)
        
    for i in range(len(device_list)):
        cur_args = copy.deepcopy(args)
        cur_args.device = int(device_list[i])
        cur_args.input = None if splits == None else list(splits)[i]
        p.apply_async(main, args=(cur_args, i, msgq, device_list), error_callback=print_subproces_run_error)

    p.close()
    p.join()
    result  = 0 if 2 * len(device_list) == msgq.qsize() else 1
    logger.info("multidevice run end qsize:{} result:{}".format(msgq.qsize(), result))
    tlist = []
    while msgq.qsize() != 0:
        ret = msgq.get()
        if type(ret) == list:
            print("i:{} device_{} throughput:{} start_time:{} end_time:{}".format(
                ret[0], device_list[ret[0]], ret[1], ret[2], ret[3]))
            tlist.append(ret[1])
    logger.info('summary throughput:{}'.format(sum(tlist)))
    return result

if __name__ == "__main__":
    args = get_args()

    version_check(args)

    if args.profiler == True:
        # try use msprof to run
        msprof_bin = shutil.which('msprof')
        if msprof_bin is None or os.getenv('GE_PROFILIGN_TO_STD_OUT') == '1':
            logger.info("find no msprof continue use acl.json mode")
        else:
            msprof_run_profiling(args)
            exit(0)

    if args.dymShape_range != None and args.dymShape is None:
        # dymshape range run,according range to run each shape infer get best shape
        dymshape_range_run(args)
        exit(0)

    if type(args.device) == list:
        # args has multiple device, run single process for each device
        ret = multidevice_run(args)
        exit(ret)

    main(args)
