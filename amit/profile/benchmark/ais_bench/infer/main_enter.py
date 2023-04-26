#
# Copyright 2023 Huawei Technologies Co., Ltd
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

import logging
import math
import os
import sys
import time
import shutil
import copy
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
from ais_bench.infer.miscellaneous import get_acl_json_path, version_check, get_batchsize, dymshape_range_run
from ais_bench.infer.args_adapter import MyArgs
from ais_bench.infer.utils import (get_fileslist_from_dir, list_split, logger)


def args_rule_apply(args:any):
    if args.profiler is True and args.dump is True:
        logger.error("parameter --profiler cannot be true at the same time as parameter --dump, please check them!\n")
        raise RuntimeError('error bad parameters --profiler and --dump')

    if (args.profiler is True or args.dump is True) and (args.output is None):
        logger.error("when dump or profiler, miss output path, please check them!")
        raise RuntimeError('miss output parameter!')

    if args.auto_set_dymshape_mode is False and args.auto_set_dymdims_mode is False:
        args.no_combine_tensor_mode = False
    else:
        args.no_combine_tensor_mode = True

    if args.profiler is True and args.warmup_count != 0 and args.input is not None:
        logger.info("profiler mode with input change warmup_count to 0")
        args.warmup_count = 0

    if args.output is None and args.output_dirname is not None:
        logger.error("parameter --output_dirname cann't be used alone."
                     " Please use it together with the parameter --output!\n")
        raise RuntimeError('error bad parameters --output_dirname')
    return args


def set_session_options(session, input_args):
    # 增加校验
    if input_args.dym_batch != 0:
        session.set_dynamic_batchsize(input_args.dym_batch)
    elif input_args.dym_hw is not None:
        hwstr = input_args.dym_hw.split(",")
        session.set_dynamic_hw((int)(hwstr[0]), (int)(hwstr[1]))
    elif input_args.dym_dims is not None:
        session.set_dynamic_dims(input_args.dym_dims)
    elif input_args.dym_shape is not None:
        session.set_dynamic_shape(input_args.dym_shape)
    else:
        session.set_staticbatch()

    if input_args.batchsize is None:
        input_args.batchsize = get_batchsize(session, input_args)
        logger.info("try get model batchsize:{}".format(input_args.batchsize))

    # 设置custom out tensors size
    if input_args.output_size is not None:
        customsizes = [int(n) for n in input_args.output_size.split(',')]
        logger.debug("set customsize:{}".format(customsizes))
        session.set_custom_outsize(customsizes)


def init_inference_session(input_args):
    acl_json_path = get_acl_json_path(input_args)
    session = InferSession(input_args.device, input_args.model, acl_json_path, input_args.debug, input_args.loop)

    set_session_options(session, input_args)
    logger.debug("session info:{}".format(session.session))
    return session


def set_dymshape_shape(session, inputs):
    dyshape_list = []
    intensors_desc = session.get_inputs()
    for i, input_i in enumerate(inputs):
        str_shape = [ str(shape) for shape in input_i.shape ]
        dyshape = "{}:{}".format(intensors_desc[i].name, ",".join(str_shape))
        dyshape_list.append(dyshape)
    dyshapes = ';'.join(dyshape_list)
    logger.debug("set dymshape shape:{}".format(dyshapes))
    session.set_dynamic_shape(dyshapes)
    summary.add_batchsize(inputs[0].shape[0])


def set_dymdims_shape(session, inputs):
    dydims_list = []
    intensors_desc = session.get_inputs()
    for i, input_i in enumerate(inputs):
        str_shape = [ str(shape) for shape in input_i.shape ]
        dydim = "{}:{}".format(intensors_desc[i].name, ",".join(str_shape))
        dydims_list.append(dydim)
    dydims = ';'.join(dydims_list)
    logger.debug("set dymdims shape:{}".format(dydims))
    session.set_dynamic_dims(dydims)
    summary.add_batchsize(inputs[0].shape[0])


def warmup(session, input_args, intensors_desc, infiles):
    # prepare input data
    infeeds = []
    for j, files in enumerate(infiles):
        if input_args.run_mode == "tensor":
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize,
                                                input_args.pure_data_type, input_args.no_combine_tensor_mode)
            infeeds.append(tensor)
        else:
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize,
                                                input_args.pure_data_type, input_args.no_combine_tensor_mode)
            infeeds.append(narray)
    session.set_loop_count(1)
    # warmup
    for _ in range(input_args.warmup_count):
        outputs = run_inference(session, input_args, infeeds, out_array=True)

    session.set_loop_count(input_args.loop)

    # reset summary info
    summary.reset()
    session.reset_sumaryinfo()
    MemorySummary.reset()
    logger.info("warm up {} done".format(input_args.warmup_count))


def run_inference(session, input_args, inputs, out_array=False):
    if input_args.auto_set_dymshape_mode:
        set_dymshape_shape(session, inputs)
    elif input_args.auto_set_dymdims_mode:
        set_dymdims_shape(session, inputs)
    outputs = session.run(inputs, out_array)
    return outputs


# tensor to loop infer
def infer_loop_tensor_run(session, input_args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference tensor Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize,
                                                input_args.pure_data_type, input_args.no_combine_tensor_mode)
            intensors.append(tensor)
        outputs = run_inference(session, input_args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(outputs, output_prefix, infiles,
                                 input_args.outfmt, i, input_args.output_batchsize_axis)


# files to loop iner
def infer_loop_files_run(session, input_args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference files Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            real_files = convert_real_files(files)
            tensor = session.create_tensor_from_fileslist(intensors_desc[j], real_files)
            intensors.append(tensor)
        outputs = run_inference(session, input_args, intensors)
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(outputs, output_prefix, infiles,
                                 input_args.outfmt, i, input_args.output_batchsize_axis)


# First prepare the data, then execute the reference, and then write the file uniformly
def infer_fulltensors_run(session, input_args, intensors_desc, infileslist, output_prefix):
    outtensors = []
    intensorslist = create_intensors_from_infileslist(infileslist, intensors_desc, session,
                                                      input_args.pure_data_type, input_args.no_combine_tensor_mode)

    #for inputs in intensorslist:
    for inputs in tqdm(intensorslist, file=sys.stdout, desc='Inference Processing full'):
        outputs = run_inference(session, input_args, inputs)
        outtensors.append(outputs)

    for i, outputs in enumerate(outtensors):
        session.convert_tensors_to_host(outputs)
        if output_prefix is not None:
            save_tensors_to_file(outputs, output_prefix, infileslist[i],
                                   input_args.outfmt, i, input_args.output_batchsize_axis)


# loop numpy array to infer
def infer_loop_array_run(session, input_args, intensors_desc, infileslist, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference array Processing')):
        innarrays = []
        for j, files in enumerate(infiles):
            narray = get_narray_from_files_list(files, intensors_desc[j].realsize, input_args.pure_data_type)
            innarrays.append(narray)
        outputs = run_inference(session, input_args, innarrays)
        session.convert_tensors_to_host(outputs)
        if input_args.output is not None:
            save_tensors_to_file(outputs, output_prefix, infiles,
                                 input_args.outfmt, i, input_args.output_batchsize_axis)


def msprof_run_profiling(input_args):
    cmd = sys.executable + " " + ' '.join(sys.argv) + " --profiler=0 --warmup_count=0"
    msprof_cmd = "{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on "\
               "--sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on "\
               "--runtime-api=on --task-time=on --aicpu=on".format(msprof_bin, input_args.output, cmd)
    logger.info("msprof cmd:{} begin run".format(msprof_cmd))
    ret = os.system(msprof_cmd)
    logger.info("msprof cmd:{} end run ret:{}".format(msprof_cmd, ret))


def main(input_args, index=0, msgq=None, device_list=None):
    # if msgq is not None,as subproces run
    if msgq is None:
        logger.info("subprocess_{} main run".format(index))

    if input_args.debug:
        logger.setLevel(logging.DEBUG)

    session = init_inference_session(input_args)

    intensors_desc = session.get_inputs()
    if device_list is not None and len(device_list) > 1:
        if input_args.output is not None:
            if input_args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(input_args.output, timestr)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            else:
                output_prefix = os.path.join(input_args.output, input_args.output_dirname)
                output_prefix = os.path.join(output_prefix, "device" + str(device_list[index]) + "_" + str(index))
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, 0o755)
            logger.info("output path:{}".format(output_prefix))
        else:
            output_prefix = None
    else:
        if input_args.output is not None:
            if input_args.output_dirname is None:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                output_prefix = os.path.join(input_args.output, timestr)
            else:
                output_prefix = os.path.join(input_args.output, input_args.output_dirname)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, 0o755)
            logger.info("output path:{}".format(output_prefix))
        else:
            output_prefix = None

    inputs_list = [] if input_args.input is None else input_args.input.split(',')

    # create infiles list accord inputs list
    if len(inputs_list) == 0:
        # Pure reference scenario. Create input zero data
        infileslist = [[ [ pure_infer_fake_file ] for index in intensors_desc ]]
    else:
        infileslist = create_infileslist_from_inputs_list(inputs_list, intensors_desc,
                                                          input_args.no_combine_tensor_mode)

    warmup(session, input_args, intensors_desc, infileslist[0])

    if msgq is not None:
		# wait subprocess init ready, if time eplapsed,force ready run
        logger.info("subprocess_{} qsize:{} now waiting".format(index, msgq.qsize()))
        msgq.put(index)
        time_sec = 0
        while True:
            if msgq.qsize() >= input_args.subprocess_count:
                break
            time_sec = time_sec + 1
            if time_sec > 10:
                logger.warning("subprocess_{} qsize:{} time:{} s elapsed".format(index, msgq.qsize(), time_sec))
                break
            time.sleep(1)
        logger.info("subprocess_{} qsize:{} ready to infer run".format(index, msgq.qsize()))

    start_time = time.time()

    if input_args.run_mode == "array":
        infer_loop_array_run(session, input_args, intensors_desc, infileslist, output_prefix)
    elif input_args.run_mode == "files":
        infer_loop_files_run(session, input_args, intensors_desc, infileslist, output_prefix)
    elif input_args.run_mode == "full":
        infer_fulltensors_run(session, input_args, intensors_desc, infileslist, output_prefix)
    elif input_args.run_mode == "tensor":
        infer_loop_tensor_run(session, input_args, intensors_desc, infileslist, output_prefix)
    else:
        raise RuntimeError('wrong run_mode:{}'.format(input_args.run_mode))

    end_time = time.time()

    summary.add_args(sys.argv)
    s = session.sumary()
    summary.npu_compute_time_list = s.exec_time_list
    summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
    summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
    summary.report(input_args.batchsize, output_prefix, input_args.display_all_summary)

    if msgq is not None:
		# put result to msgq
        msgq.put([index, summary.infodict.get('throughput'), start_time, end_time])

    session.finalize()


def print_subproces_run_error(value):
    logger.error("subprocess run failed error_callback:{}".format(value))


def seg_input_data_for_multi_process(input_args, inputs, jobs):
    inputs_list = [] if inputs is None else inputs.split(',')
    if inputs_list is None:
        return inputs_list

    fileslist = []
    if os.path.isfile(inputs_list[0]):
        fileslist = inputs_list
    elif os.path.isdir(inputs_list[0]):
        for dir_name in inputs_list:
            fileslist.extend(get_fileslist_from_dir(dir_name))
    else:
        logger.error('error {} not file or dir'.format(inputs_list[0]))
        raise RuntimeError()

    input_args.device = 0
    session = init_inference_session(input_args)
    intensors_desc = session.get_inputs()
    try:
        chunks_elements = math.ceil(len(fileslist) / len(intensors_desc))
    except ZeroDivisionError as e:
        logger.error('Incorrect model input desc.(The model input desc is 0)')
        raise e
    chunks = list(list_split(fileslist, chunks_elements, None))
    fileslist = [ [] for e in range(jobs) ]
    for chunk in chunks:
        try:
            splits_elements = math.ceil(len(chunk) / jobs)
        except ZeroDivisionError as e:
            logger.error('Incorrect model input desc.(The model input desc is 0)')
            raise e
        splits = list(list_split(chunk, splits_elements, None))
        for j, split in enumerate(splits):
            fileslist[j].extend(split)
    res = []
    for files in fileslist:
        res.append(','.join(list(filter(None, files))))
    return res


def multidevice_run(input_args):
    logger.info("multidevice:{} run begin".format(input_args.device))
    device_list = input_args.device
    p = Pool(len(device_list))
    msgq = Manager().Queue()

    input_args.subprocess_count = len(device_list)
    jobs = input_args.subprocess_count
    splits = None
    if (input_args.input is not None):
        splits = seg_input_data_for_multi_process(input_args, input_args.input, jobs)

    for i, device in enumerate(device_list):
        cur_args = copy.deepcopy(input_args)
        cur_args.device = int(device)
        cur_args.input = None if splits is None else list(splits)[i]
        p.apply_async(main, args=(cur_args, i, msgq, device_list), error_callback=print_subproces_run_error)

    p.close()
    p.join()
    result = 0 if 2 * len(device_list) == msgq.qsize() else 1
    logger.info("multidevice run end qsize:{} result:{}".format(msgq.qsize(), result))
    tlist = []
    while msgq.qsize() != 0:
        ret = msgq.get()
        if type(ret) == list:
            logger.info("i:{} device_{} throughput:{} start_time:{} end_time:{}".format(
                ret[0], device_list[ret[0]], ret[1], ret[2], ret[3]))
            tlist.append(ret[1])
    logger.info('summary throughput:{}'.format(sum(tlist)))
    return result


def main_enter(args:MyArgs):
    args = args_rule_apply(args)

    version_check(args)

    if args.profiler is True:
        # try use msprof to run
        msprof_bin = shutil.which('msprof')
        if msprof_bin is None or os.getenv('GE_PROFILIGN_TO_STD_OUT') == '1':
            logger.info("find no msprof continue use acl.json mode")
        else:
            msprof_run_profiling(args)
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