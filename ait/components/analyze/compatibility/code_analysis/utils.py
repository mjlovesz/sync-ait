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
import json


def get_valid_read_file(file_path):
    real_file_path = os.path.realpath(file_path)
    if os.path.islink(real_file_path):
        raise PermissionError('Opening softlink file is not permitted.')
    if not os.path.isfile(real_file_path):
        raise ValueError(f'Provided file_path={file_path} is not a file.')
    if not os.access(real_file_path, os.R_OK):
        raise PermissionError(f'Opening file_path={file_path} is not permitted.')
    return real_file_path


def get_data(filename, dir_path='.', second_path=''):
    file_path = os.path.join(dir_path, second_path, filename)
    real_file_path = get_valid_read_file(file_path)

    with open(real_file_path, 'r') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data


# 检查../../data/profiling目录中是否存在profiling文件，并检查该profiling文件是否正确配置，返回profiling文件路径
# profiling目录中只能存在一个profiling文件
def check_profiling_data(datapath):
    profiling_nums = 0
    for file in os.listdir(datapath):
        if (file[0:5] == "PROF_"):
            profiling_nums += 1
    if profiling_nums == 0:
        raise Exception(
            f"profiling data do not in {datapath},or the file name is incorrect." \
                "Use the original name, such as PROF_xxxxx"
        )
    elif profiling_nums > 1:
        raise Exception(
            "The number of profiling data is greater than 1, " \
                "Please enter only one profiling data"
        )
    datapath = os.path.join(datapath, os.listdir(datapath)[0])
    filename_is_correct = 1
    for file in os.listdir(datapath):
        if (file[0:7] == 'device_'):
            filename_is_correct = 0
    if filename_is_correct:
        raise ValueError(
            f'{datapath} is not a correct profiling file, \
                correct profiling file is PROF_xxxxxxxx and it includes device_*')
    return datapath


def get_statistic_profile_data(profile_path):
    for device in os.listdir(profile_path):
        path = os.path.join(profile_path, device, "summary")
        for file in os.listdir(path):
            if "acl_statistic" in file:
                acl_statistic_path = os.path.join(path, file)
    return open(acl_statistic_path)


def get_profile_data(profilepath):
    profilepath = os.path.join(profilepath, os.listdir(profilepath)[0])
    return profilepath

