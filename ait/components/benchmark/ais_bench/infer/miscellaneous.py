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
import sys
import stat
import json
import itertools
import numpy as np

from ais_bench.infer.utils import logger
from components.utils.file_open import ms_open, MAX_SIZE_LIMITE_CONFIG_FILE, MAX_SIZE_LIMITE_NORMAL_FILE

PERMISSION_DIR = 0o750

ACL_JSON_CMD_LIST = [
    "output",
    "storage_limit",
    "ascendcl",
    "runtime_api",
    "hccl",
    "task_time",
    "aicpu",
    "aic_metrics",
    "l2",
    "sys_hardware_mem_freq",
    "lcc_profiling",
    "dvpp_freq",
    "host_sys",
    "host_sys_usage",
    "host_sys_usage_freq",
    "sys_interconnection_freq",
    "msproftx"
]


def get_modules_version(name):
    try:
        import pkg_resources
    except ImportError as err:
        raise Exception("importerror") from err
    pkg = pkg_resources.get_distribution(name)
    return pkg.version


def version_check(args):
    try:
        aclruntime_version = get_modules_version('aclruntime')
    except Exception:
        url = 'https://gitee.com/ascend/tools.git'
        logger.warning(f"can't find aclruntime, please visit {url} to install ais_bench(benchmark)"
                       "to install")
        args.run_mode = "tensor"
    if (aclruntime_version != "0.0.2"):
        logger.warning(f"aclruntime{aclruntime_version} version is lower please update \
                        aclruntime follow any one method")
        # set old run mode to run ok
        args.run_mode = "tensor"


def get_model_name(model):
    path_list = model.split('/')
    return path_list[-1][:-3]


def check_valid_acl_json_for_dump(acl_json_path, model):
    with ms_open(acl_json_path, mode="r", max_size=MAX_SIZE_LIMITE_CONFIG_FILE) as f:
        acl_json_dict = json.load(f)
    model_name_correct = get_model_name(model)
    if acl_json_dict.get("dump") is not None:
        # check validity of dump_list (model_name)
        dump_list_val = acl_json_dict["dump"].get("dump_list")
        if dump_list_val is not None:
            if dump_list_val == [] or dump_list_val[0].get("model_name") != model_name_correct:
                logger.warning("dump failed, 'model_name' is not set or set incorrectly. correct"
                               "'model_name' should be {}".format(model_name_correct))
        else:
            logger.warning("dump failed, acl.json need to set 'dump_list' attribute")
        # check validity of dump_path
        dump_path_val = acl_json_dict["dump"].get("dump_path")
        if dump_path_val is not None:
            if os.path.isdir(dump_path_val) and os.access(dump_path_val, os.R_OK) and os.access(dump_path_val, os.W_OK):
                pass
            else:
                logger.warning("dump failed, 'dump_path' not exists or has no read/write permission")
        else:
            logger.warning("dump failed, acl.json need to set 'dump_path' attribute")
        # check validity of dump_op_switch
        dump_op_switch_val = acl_json_dict["dump"].get("dump_op_switch")
        if dump_op_switch_val is not None and dump_op_switch_val not in {"on", "off"}:
            logger.warning("dump failed, 'dump_op_switch' need to be set as 'on' or 'off'")
        # check validity of dump_mode
        dump_mode_val = acl_json_dict["dump"].get("dump_mode")
        if dump_mode_val is not None and dump_mode_val not in {"input", "output", "all"}:
            logger.warning("dump failed, 'dump_mode' need to be set as 'input', 'output' or 'all'")
    return


def get_acl_json_path(args):
    """
    get acl json path. when args.profiler is true or args.dump is True, create relative acl.json ,
    default current folder
    """
    if args.acl_json_path is not None:
        check_valid_acl_json_for_dump(args.acl_json_path, args.model)
        return args.acl_json_path
    if not args.profiler and not args.dump:
        return None

    output_json_dict = {}
    if args.profiler:
        output_json_dict = {"profiler": {"switch": "on", "aicpu": "on", "output": "", "aic_metrics": ""}}
        out_profiler_path = os.path.join(args.output, "profiler")

        if not os.path.exists(out_profiler_path):
            os.makedirs(out_profiler_path, PERMISSION_DIR)
        output_json_dict["profiler"]["output"] = out_profiler_path
    elif args.dump:
        output_json_dict = {"dump": {"dump_path": "", "dump_mode": "all", "dump_list": [{"model_name": ""}]}}
        out_dump_path = os.path.join(args.output, "dump")

        if not os.path.exists(out_dump_path):
            os.makedirs(out_dump_path, PERMISSION_DIR)

        model_name = args.model.split("/")[-1]
        output_json_dict["dump"]["dump_path"] = out_dump_path
        output_json_dict["dump"]["dump_list"][0]["model_name"] = model_name.split('.')[0]

    out_json_file_path = os.path.join(args.output, "acl.json")

    OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR
    with ms_open(out_json_file_path, mode="w") as f:
        json.dump(output_json_dict, f, indent=4, separators=(", ", ": "), sort_keys=True)
    return out_json_file_path


def get_batchsize(session, args):
    intensors_desc = session.get_inputs()
    batchsize = intensors_desc[0].shape[0]
    if args.dym_batch != 0:
        batchsize = int(args.dym_batch)
    elif args.dym_dims is not None or args.dym_shape is not None:
        instr = args.dym_dims if args.dym_dims is not None else args.dym_shape
        elems = instr.split(';')
        for elem in elems:
            name, shapestr = elem.split(':')
            if name == intensors_desc[0].name:
                batchsize = int(shapestr.split(',')[0])
    return batchsize


def get_range_list(ranges):
    elems = ranges.split(';')
    info_list = []
    for elem in elems:
        shapes = []
        name, shapestr = elem.split(':')
        for content in shapestr.split(','):
            step = 1
            if '~' in content:
                start = int(content.split('~')[0])
                end = int(content.split('~')[1])
                step = int(content.split('~')[2]) if len(content.split('~')) == 3 else 1
                ranges = [ str(i) for i in range(start, end+1, step)]
            elif '-' in content :
                ranges = content.split('-')
            else:
                start = int(content)
                ranges = [ str(start) ]
            shapes.append(ranges)
            logger.debug("content:{} get range{}".format(content, ranges))
        shape_list = [ ','.join(s) for s in list(itertools.product(*shapes)) ]
        info = [ "{}:{}".format(name, s) for s in shape_list ]
        info_list.append(info)
        logger.debug("name:{} shapes:{} info:{}".format(name, shapes, info))

    res = [ ';'.join(s) for s in list(itertools.product(*info_list)) ]
    logger.debug("range list:{}".format(res))
    return res


# get dymshape list from input_ranges
# input_ranges can be a string like "name1:1,3,224,224;name2:1,600" or file
def get_dymshape_list(input_ranges):
    ranges_list = []
    if os.path.isfile(input_ranges):
        with ms_open(input_ranges, mode="rt", max_size=MAX_SIZE_LIMITE_NORMAL_FILE, encoding='utf-8') as finfo:
            line = finfo.readline()
            while line:
                line = line.rstrip('\n')
                ranges_list.append(line)
                line = finfo.readline()
    else:
        ranges_list.append(input_ranges)

    dymshape_list = []
    for ranges in ranges_list:
        dymshape_list.extend(get_range_list(ranges))
    return dymshape_list


# get throughput from out log
def get_throughtput_from_log(log_path):
    if not os.path.exists(log_path):
        return "Failed", 0
    cmd = "cat {} | grep throughput".format(log_path)
    cmd = cmd + " | awk '{print $NF}'"
    try:
        outval = os.popen(cmd).read()
    except Exception as e:
        logger.warning("get throughtput failed e:{}".format(e))
        return "Failed", 0
    if outval == "":
        return "Failed", 0
    throughtput = float(outval)
    return "OK", throughtput


def dymshape_range_run(args):
    dymshape_list = get_dymshape_list(args.dym_shape_range)
    results = []
    log_path = os.path.realpath("dym.log") if args.output is None else os.path.join(args.output, "dym.log")
    for dymshape in dymshape_list:
        cmd = "rm -rf {};{} {} {}".format(log_path, sys.executable, ' '.join(sys.argv),
            "--dym-shape={}  | tee {}".format(dymshape,  log_path))
        result = { "dymshape" : dymshape, "cmd": cmd, "result": "Failed", "throughput" : 0 }
        logger.debug("cmd:{}".format(cmd))
        os.system(cmd)
        result["result"], result["throughput"] = get_throughtput_from_log(log_path)
        logger.info("dymshape:{} end run result:{}".format(dymshape, result["result"]))
        results.append(result)

    tlist = [ result["throughput"] for result in results if result["result"] == "OK" ]
    logger.info("-----------------dyshape_range Performance Summary------------------")
    logger.info("run_count:{} success_count:{} avg_throughput:{}".format(
        len(results), len(tlist), np.mean(tlist)))
    results.sort(key=lambda x: x['throughput'], reverse=True)
    for i, result in enumerate(results):
        logger.info("{} dymshape:{}  result:{} throughput:{}".format(
            i, result["dymshape"], result["result"], result["throughput"]))
    logger.info("------------------------------------------------------")
