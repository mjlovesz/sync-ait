# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import stat
import re
import uuid
from pickle import NONE
import logging
from random import sample
from string import digits, ascii_uppercase, ascii_lowercase
import json
import shutil
import shlex
import subprocess
import numpy as np
from ais_bench.infer.path_security_check import (
    ms_open,
    MAX_SIZE_LIMITE_NORMAL_FILE,
    MAX_SIZE_LIMITE_CONFIG_FILE,
    FileStat,
    is_legal_args_path_string,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PERMISSION_DIR = 0o750
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
MSACCUCMP_FILE_PATH = "tools/operator_cmp/compare/msaccucmp.py"
CANN_PATH = "/usr/local/Ascend/ascend-toolkit/latest"


# Split a List Into Even Chunks of N Elements
def list_split(list_a, n, padding_file):
    for x in range(0, len(list_a), n):
        every_chunk = list_a[x : n + x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + [padding_file for _ in range(n - len(every_chunk))]
        yield every_chunk


def list_share(list_a, count, num, left):
    head = 0
    for i in range(count):
        if i < left:
            every_chunk = list_a[head : head + num + 1]
            head = head + num + 1
        else:
            every_chunk = list_a[head : head + num]
            head = head + num
        yield every_chunk


def natural_sort(lst):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)


def get_fileslist_from_dir(dir_):
    files_list = []

    for f in os.listdir(dir_):
        f_true_path = os.path.join(dir_, f)
        f_stat = FileStat(f_true_path)
        if not f_stat.is_basically_legal('read'):
            raise RuntimeError(f'input data:{f_true_path} is illegal')
        if f_stat.is_dir:
            continue
        if f.endswith(".npy") or f.endswith(".NPY") or f.endswith(".bin") or f.endswith(".BIN"):
            files_list.append(os.path.join(dir_, f))

    if len(files_list) == 0:
        logger.error('{} of input args not find valid file,valid file format:[*.npy *.NPY *.bin *.BIN]'.format(dir_))
        raise RuntimeError()
    files_list.sort()
    return natural_sort(files_list)


def get_file_datasize(file_path):
    if file_path.endswith(".NPY") or file_path.endswith(".npy"):
        ndata = np.load(file_path)
        return ndata.nbytes
    else:
        return os.path.getsize(file_path)


def get_file_content(file_path):
    if file_path.endswith(".NPY") or file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        with ms_open(file_path, mode="rb", max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as fd:
            barray = fd.read()
            return np.frombuffer(barray, dtype=np.int8)


def get_ndata_fmt(ndata):
    if ndata.dtype == np.float32 or ndata.dtype == np.float16 or ndata.dtype == np.float64:
        fmt = "%f"
    else:
        fmt = "%d"
    return fmt


def save_data_to_files(file_path, ndata):
    if file_path.endswith(".NPY") or file_path.endswith(".npy"):
        with ms_open(file_path, mode="wb") as f:
            np.save(f, ndata)
    elif file_path.endswith(".TXT") or file_path.endswith(".txt"):
        outdata = ndata.reshape(-1, ndata.shape[-1])
        fmt = get_ndata_fmt(outdata)
        with ms_open(file_path, mode="wb") as f:
            for i in range(outdata.shape[0]):
                np.savetxt(f, np.c_[outdata[i]], fmt=fmt, newline=" ")
                f.write(b"\n")
    else:
        with ms_open(file_path, mode="wb") as f:
            ndata.tofile(f)


def create_fake_file_name(pure_data_type, index):
    suffix = "_" + pure_data_type + "_" + str(index)
    loop_max = 1000
    for _ in range(loop_max):
        fname = os.path.join(os.getcwd(), "tmp-" + "".join(str(uuid.uuid4())) + suffix)
        if not os.path.exists(fname):
            return fname
    raise RuntimeError(f'create_fake_file_name failed: inner error')


def get_dump_relative_paths(output_dir, timestamp):
    if output_dir is None or timestamp is None:
        return []
    dump_dir = os.path.join(output_dir, timestamp)
    dump_relative_paths = []
    for subdir, _, files in os.walk(dump_dir):
        if len(files) > 0:
            dump_relative_paths.append(os.path.relpath(subdir, dump_dir))
    return dump_relative_paths


def get_msaccucmp_path():
    ascend_toolkit_path = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not is_legal_args_path_string(ascend_toolkit_path):
        raise TypeError(f"ASCEND_TOOLKIT_HOME:{ascend_toolkit_path} is illegal")
    if ascend_toolkit_path is None:
        ascend_toolkit_path = CANN_PATH
    msaccucmp_path = os.path.join(ascend_toolkit_path, MSACCUCMP_FILE_PATH)
    return msaccucmp_path if os.path.exists(msaccucmp_path) else None


def make_dirs(path):
    ret = 0
    if not os.path.exists(path):
        try:
            os.makedirs(path, PERMISSION_DIR)
        except Exception as e:
            logger.warning(f"make dir {path} failed")
            ret = -1
    return ret


def create_tmp_acl_json(acl_json_path):
    with ms_open(acl_json_path, mode="r", max_size=MAX_SIZE_LIMITE_CONFIG_FILE) as f:
        acl_json_dict = json.load(f)
    tmp_acl_json_path, real_dump_path, tmp_dump_path = None, None, None

    # create tmp acl.json path
    acl_json_path_list = acl_json_path.split("/")
    acl_json_path_list[-1] = str(uuid.uuid4()) + "_" + acl_json_path_list[-1]
    tmp_acl_json_path = "/".join(acl_json_path_list)

    # change acl_json_dict
    if acl_json_dict.get("dump") is not None and acl_json_dict["dump"].get("dump_path") is not None:
        real_dump_path = acl_json_dict["dump"]["dump_path"]
        dump_path_list = real_dump_path.split("/")
        if dump_path_list[-1] == "":
            dump_path_list.pop()
        dump_path_list.append(str(uuid.uuid4()))
        tmp_dump_path = "/".join(dump_path_list)
        acl_json_dict["dump"]["dump_path"] = tmp_dump_path
        if make_dirs(tmp_dump_path) != 0:
            tmp_dump_path = None
            os.remove(tmp_acl_json_path)
            tmp_acl_json_path = None

    if tmp_acl_json_path is not None:
        with ms_open(tmp_acl_json_path, mode="w") as f:
            json.dump(acl_json_dict, f)

    return tmp_acl_json_path, real_dump_path, tmp_dump_path


def convert_helper(output_dir, timestamp):  # convert bin file in src path and output the npy file in dest path
    '''
    before:
    output_dir--|--2023***2--...  (原来可能存在的时间戳路径)
                |--2023***3--...  (原来可能存在的时间戳路径)
                |--timestamp--...  (移动过的bin file目录)

    after:
    output_dir--|--2023***2--...  (原来可能存在的时间戳路径)
                |--2023***3--...  (原来可能存在的时间戳路径)
                |--timestamp--...  (移动过的bin file目录)
                |--timestamp_npy--...  (转换后npy保存的目录)
    '''
    dump_relative_paths = get_dump_relative_paths(output_dir, timestamp)
    msaccucmp_path = get_msaccucmp_path()
    python_path = sys.executable
    if python_path is None:
        logger.error("convert_helper failed: python executable is not found. NPY file transfer failed.")
        return
    if msaccucmp_path is None:
        logger.error("convert_helper failed: msaccucmp.py is not found. NPY file transfer failed.")
        return
    if dump_relative_paths == []:
        logger.error("convert_helper failed: dump_relative_paths is empty. NPY file transfer failed.")
        return
    for dump_relative_path in dump_relative_paths:
        dump_npy_path = os.path.join(output_dir, timestamp + "_npy", dump_relative_path)
        real_dump_path = os.path.join(output_dir, timestamp, dump_relative_path)
        convert_cmd = f"{python_path} {msaccucmp_path} convert -d {real_dump_path} -out {dump_npy_path}"
        convert_cmd_list = shlex.split(convert_cmd)
        ret = subprocess.call(convert_cmd_list, shell=False)
        if ret != 0:
            logger.error(f"convert_helper failed: cmd {convert_cmd} execute failed")


def move_subdir(src_dir, dest_dir):
    # move the subdir in src_dir to dest_dir return dest_dir/subdir
    # and remove the src_dir
    '''
    before:
    src_dir--2023***1--...  (bin file存在的路径)

    dest_dir--|--2023***2--...  (原来可能存在的时间戳路径)
              |--2023***3--...  (原来可能存在的时间戳路径)

    after:
    dest_dir--|--2023***2--...  (原来可能存在的时间戳路径)
              |--2023***3--...  (原来可能存在的时间戳路径)
              |--2023***1--...  (bin file移动到新的目录下)
    '''
    res_dest, res_subdir = None, None
    subdirs = os.listdir(src_dir)
    if len(subdirs) != 1:
        logger.error(
            "move_subdir failed: multiple or none directory under src dir %s. " "The reason might be dump failed.",
            src_dir,
        )
    else:
        if os.path.exists(os.path.join(dest_dir, subdirs[0])):
            logger.error("move_subdir failed: dest dir %s exists" % os.path.join(dest_dir, subdirs[0]))
        else:
            shutil.move(os.path.join(src_dir, subdirs[0]), os.path.join(dest_dir, subdirs[0]))
            res_dest, res_subdir = dest_dir, subdirs[0]
    return res_dest, res_subdir
