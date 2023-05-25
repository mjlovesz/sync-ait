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
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR


# Split a List Into Even Chunks of N Elements
def list_split(list_a, n, padding_file):
    for x in range(0, len(list_a), n):
        every_chunk = list_a[x: n+x]

        if len(every_chunk) < n:
            every_chunk = every_chunk + \
                [padding_file for _ in range(n-len(every_chunk))]
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
