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
import logging
import stat
import json
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE

OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


def get_times_list(file):
    time_list = []
    with ms_open(file, mode="rb", max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as f:
        for line in f.readlines():
            s = line[0:-3]
            value = float(s)
            time_list.append(value)
    return time_list


def get_pid(file):
    pid = None
    if not os.path.exists(file):
        logging.info(f"{file} file not exist")
    else:
        with ms_open(file, mode="rb", max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as fd:
            pid = int(fd.read())
    return pid


if __name__ == '__main__':
    times_file = sys.argv[1]
    pid_file = sys.argv[2]
    out_file = sys.argv[3]

    times = get_times_list(times_file)
    t_pid = get_pid(pid_file)
    info = {"pid": t_pid, "npu_compute_time_list": times}
    with ms_open(out_file, mode="w") as ff:
        json.dump(info, ff)

