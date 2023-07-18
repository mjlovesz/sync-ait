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
from pickle import NONE
import logging
import json
import shutil
import numpy as np
import uuid

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
MSACCUCMP_FILE_PATH =  "tools/operator_cmp/compare/msaccucmp.py"


# Split a List Into Even Chunks of N Elements
def list_split(list_a, n, padding_file):
    for x in range(0, len(list_a), n):
        every_chunk = list_a[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [padding_file for _ in range(n-len(every_chunk))]
        yield every_chunk


def list_share(list_a, count, num, left):
    head = 0
    for i in range(count):
        if i < left:
            every_chunk = list_a[head: head + num + 1]
            head = head + num + 1
        else:
            every_chunk = list_a[head: head + num]
            head = head + num
        yield every_chunk


def natural_sort(lst):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(lst, key=alphanum_key)


def get_fileslist_from_dir(dir_):
    files_list = []

    if os.path.exists(dir_) is False:
        logger.error('dir:{} not exist'.format(dir_))
        raise RuntimeError()

    for f in os.listdir(dir_):
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
        with open(file_path, 'rb') as fd:
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
        np.save(file_path, ndata)
    elif file_path.endswith(".TXT") or file_path.endswith(".txt"):
        outdata = ndata.reshape(-1, ndata.shape[-1])
        fmt = get_ndata_fmt(outdata)
        with os.fdopen(os.open(file_path, WRITE_FLAGS, WRITE_MODES), 'wb') as f:
            for i in range(outdata.shape[0]):
                np.savetxt(f, np.c_[outdata[i]], fmt=fmt, newline=" ")
                f.write(b"\n")
    else:
        ndata.tofile(file_path)


def get_dump_relative_paths(output_dir, timestamp):
    dump_dir = os.path.join(output_dir, timestamp)
    dump_relative_paths = []
    for subdir, _, files in os.walk(dump_dir):
        if len(files) > 0:
            dump_relative_paths.append((dump_dir, os.path.relpath(subdir, dump_dir)))
    return dump_relative_paths


def get_msaccucmp_path():
    ascend_toolkit_path = os.environ.get("ASCEND_TOOLKIT_HOME")
    if ascend_toolkit_path is None:
        return None
    msaccucmp_path = os.path.join(str(ascend_toolkit_path), MSACCUCMP_FILE_PATH)
    return msaccucmp_path if os.path.exists(msaccucmp_path) else None




def create_tmp_acl_json(acl_json_path):
    with open(acl_json_path, 'r') as f:
        acl_json_dict = json.load(f)
    tmp_acl_json_path, real_dump_path, tmp_dump_path = None, None, None

    # create tmp acl.json path
    acl_json_path_list = acl_json_path.split("/")
    acl_json_path_list[-1] = str(uuid.uuid4()) + acl_json_path_list[-1]
    tmp_acl_json_path = "/".join(acl_json_path_list)

    # change acl_json_dict
    if acl_json_dict.get("dump") is not None:
        real_dump_path = acl_json_dict["dump"].get("dump_path")
        if real_dump_path is not None:
            dump_path_list = real_dump_path.split("/")
            if dump_path_list[-1] == "":
                dump_path_list.pop()
            dump_path_list.append(str(uuid.uuid4()))
            tmp_dump_path = "/".join(dump_path_list)
            acl_json_dict["dump"]["dump_path"] = tmp_dump_path

    if tmp_acl_json_path is not None:
        with open(tmp_acl_json_path, "w") as f:
            json.dump(acl_json_dict, f)

    return tmp_acl_json_path, real_dump_path, tmp_dump_path


def convert(output_dir, timestamp): # convert bin file in src path and output the npy file in dest path
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
    dump_relative_paths = get_dump_relative_paths(output_dir, timestamp) # find dump dir in output_path and return the lastest timestamp dir
    msaccucmp_path = get_msaccucmp_path()
    python_path = sys.executable
    if python_path is not None and dump_relative_paths != [] and msaccucmp_path is not None:
        for dump_relative_path in dump_relative_paths:
            dump_npy_path = os.path.join(output_dir, timestamp + "_npy", dump_relative_path)
            real_dump_path = os.path.join(output_dir, timestamp, dump_relative_path)
            cmd = f"{python_path} {msaccucmp_path} convert -d {real_dump_path} -out {dump_npy_path}"
            ret = os.system(cmd)
            if ret != 0:
                logger.warning(f"convert failed: cannot convert binfiles in {real_dump_path} to {dump_npy_path}")


def transfer_remove(src_dir, dest_dir): 
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
    subdirs = os.listdir(src_dir)
    if len(subdirs) != 1:
        return None
    shutil.move(os.path.join(src_dir, subdirs[0]), os.path.join(dest_dir, subdirs[0]))
    os.rmdir(src_dir)
    return dest_dir, subdirs[0]
