import logging
import math
import os
import sys
import time
import shutil
import copy
import re
import subprocess
from multiprocessing import Pool
from multiprocessing import Manager

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
from ais_bench.infer.args_adapter import BenchMarkArgsAdapter
from ais_bench.infer.backends import BackendFactory

def set_session_options(session, args):
    # 增加校验
    aipp_batchsize = -1
    if args.dym_batch != 0:
        session.set_dynamic_batchsize(args.dym_batch)
        aipp_batchsize = session.get_max_dym_batchsize()
    elif args.dym_hw is not None:
        hwstr = args.dym_hw.split(",")
        session.set_dynamic_hw((int)(hwstr[0]), (int)(hwstr[1]))
    elif args.dym_dims is not None:
        session.set_dynamic_dims(args.dym_dims)
    elif args.dym_shape is not None:
        session.set_dynamic_shape(args.dym_shape)
    else:
        session.set_staticbatch()

    if args.batchsize is None:
        args.batchsize = get_batchsize(session, args)
        logger.info("try get model batchsize:{}".format(args.batchsize))

    if aipp_batchsize < 0:
        aipp_batchsize = args.batchsize

    # 确认模型只有一个动态 aipp input
    aipp_input_exsity = session.get_dym_aipp_input_exsity()
    if (args.aipp_config is not None) and (aipp_input_exsity == 1):
        session.load_aipp_config_file(args.aipp_config, aipp_batchsize)
        session.check_dym_aipp_input_exsity()
    elif (args.aipp_config is None) and (aipp_input_exsity == 1):
        logger.error("can't find aipp config file for model with dym aipp input , please check it!")
        raise RuntimeError('aipp model without aipp config!')
    elif (aipp_input_exsity > 1):
        logger.error("don't support more than one dynamic aipp input in model, amount of aipp input is {}"
                        .format(aipp_input_exsity))
        raise RuntimeError('aipp model has more than 1 aipp input!')
    elif (aipp_input_exsity == -1):
        raise RuntimeError('aclmdlGetAippType failed!')

    # 设置custom out tensors size
    if args.output_size is not None:
        customsizes = [int(n) for n in args.output_size.split(',')]
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
    if args.auto_set_dymshape_mode is True:
        set_dymshape_shape(session, inputs)
    elif args.auto_set_dymdims_mode is True:
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
        if output_prefix is not None:
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
        if output_prefix is not None:
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
        if output_prefix is not None:
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
        if args.output is not None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i, args.output_batchsize_axis)


def msprof_run_profiling(args, msprof_bin):
    cmd = sys.executable + " " + ' '.join(sys.argv) + " --profiler=0 --warmup-count=0"
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
        msprof_bin, args.output, cmd)
    logger.info("msprof cmd:{} begin run".format(msprof_cmd))
    ret = os.system(msprof_cmd)
    logger.info("msprof cmd:{} end run ret:{}".format(msprof_cmd, ret))


def get_energy_consumption(npu_id):
    cmd = "npu-smi info -t power -i {}".format(npu_id)
    get_npu_id = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    npu_id = get_npu_id.stdout.decode('gb2312')
    power = []
    npu_id = npu_id.split("\n")
    for key in npu_id:
        if key.find("Power Dissipation(W)", 0, len(key)) != -1:
            power = key[34:len(key)]
            break

    return power


def main(args, index=0, msgq=None, device_list=None):
    # if msgq is not None,as subproces run
    if msgq is not None:
        logger.info("subprocess_{} main run".format(index))

    if args.debug is True:
        logger.setLevel(logging.DEBUG)

    session = init_inference_session(args)

    intensors_desc = session.get_inputs()
    if device_list is not None and len(device_list) > 1:
        if args.output is not None:
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
        if args.output is not None:
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

    if msgq is not None:
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
    start_energy_consumption = 0
    end_energy_consumption = 0
    if args.energy_consumption:
        start_energy_consumption = get_energy_consumption(args.npu_id)

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
    if args.energy_consumption:
        end_energy_consumption = get_energy_consumption(args.npu_id)
    end_time = time.time()

    summary.add_args(sys.argv)
    s = session.sumary()
    summary.npu_compute_time_list = s.exec_time_list
    summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
    summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
    summary.report(args.batchsize, output_prefix, args.display_all_summary)
    if args.energy_consumption:
        logger.info("npu_id {} energy_consumption {}".format(args.npu_id, (float(end_energy_consumption) -
                    float(start_energy_consumption)) * (end_time - start_time)))
    if msgq is not None:
		# put result to msgq
        msgq.put([index, summary.infodict['throughput'], start_time, end_time])

    session.finalize()


def print_subproces_run_error(value):
    logger.error("subprocess run failed error_callback:{}".format(value))


def seg_input_data_for_multi_process(args, inputs, jobs):
    inputs_list = [] if inputs is None else inputs.split(',')
    if inputs_list is None:
        return inputs_list

    fileslist = []
    if os.path.isfile(inputs_list[0]) is True:
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
    npu_id_list = args.npu_id
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
        if args.energy_consumption:
            cur_args.npu_id = int(npu_id_list[i])
        cur_args.input = None if splits is None else list(splits)[i]
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


def args_rules(args):
    if args.profiler is True and args.dump is True:
        logger.error("parameter --profiler cannot be true at the same time as parameter --dump, please check them!\n")
        raise RuntimeError('error bad parameters --profiler and --dump')

    if (args.profiler is True or args.dump is True) and (args.output is None):
        logger.error("when dump or profiler, miss output path, please check them!")
        raise RuntimeError('miss output parameter!')

    # 判断--aipp_config 文件是否是存在的.config文件
    if (args.aipp_config is not None):
        if (os.path.splitext(args.aipp_config)[-1] == ".config"):
            if (os.path.isfile(args.aipp_config) is not True):
                logger.error("can't find the path of config file, please check it!")
                raise RuntimeError('wrong aipp config file path!')
        else:
            logger.error("aipp config file is not a .config file, please check it!")
            raise RuntimeError('wrong aipp config file type!')

    if args.auto_set_dymshape_mode == False and args.auto_set_dymdims_mode == False:
        args.no_combine_tensor_mode = False
    else:
        args.no_combine_tensor_mode = True

    if args.profiler is True and args.warmup_count != 0 and args.input != None:
        logger.info("profiler mode with input change warmup_count to 0")
        args.warmup_count = 0

    if args.output is None and args.output_dirname is not None:
        logger.error("parameter --output_dirname cann't be used alone. Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output_dirname')
    return args


def backend_run(args):
    backend_class = BackendFactory.create_backend(args.backend)
    backend = backend_class(args)
    backend.load(args.model)
    backend.run()
    perf = backend.get_perf()
    logger.info("perf info:{}".format(perf))


def benchmark_process(args:BenchMarkArgsAdapter):
    args = args_rules(args)
    version_check(args)

    if args.perf is True:
        backend_run(args)
        return 0

    if args.profiler is True:
        # try use msprof to run
        msprof_bin = shutil.which('msprof')
        if msprof_bin is None or os.getenv('GE_PROFILIGN_TO_STD_OUT') == '1':
            logger.info("find no msprof continue use acl.json mode")
        else:
            msprof_run_profiling(args, msprof_bin)
            return 0

    if args.dym_shape_range is not None and args.dym_shape is None:
        # dymshape range run,according range to run each shape infer get best shape
        dymshape_range_run(args)
        return 0

    if type(args.device) == list:
        # args has multiple device, run single process for each device
        ret = multidevice_run(args)
        return ret

    main(args)
    return 0
