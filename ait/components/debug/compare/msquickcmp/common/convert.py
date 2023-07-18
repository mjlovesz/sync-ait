# coding=utf-8
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

from msquickcmp.common import utils


MSACCUCMP_FILE_PATH =  "toolkit/tools/operator_cmp/compare/msaccucmp.py"


def convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, cann_path):
    """
    Function Description:
        provide the interface for dump data conversion
    Exception Description:
        when invalid msaccucmp command throw exception
    """
    common_path = os.path.commonprefix([npu_dump_data_path, npu_net_output_data_path])
    npu_dump_data_path_diff = os.path.relpath(npu_dump_data_path, common_path)
    time_stamp_file_path = npu_dump_data_path_diff.split(os.path.sep)[1]
    convert_dir_path = npu_dump_data_path.replace(time_stamp_file_path, time_stamp_file_path+'_bin2npy')
    convert_dir_path = os.path.normpath(convert_dir_path)
    if not os.path.exists(convert_dir_path):
            os.makedirs(convert_dir_path)
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    python_version = sys.executable.split('/')[-1]
    bin2npy_cmd = [python_version, msaccucmp_command_file_path,
                    "convert", "-d", npu_dump_data_path, "-out", convert_dir_path]
    utils.execute_command(bin2npy_cmd)
    utils.logger.info("msaccucmp command line: %s " % " ".join(bin2npy_cmd))
    return convert_dir_path


def convert_bin_file_to_npy(bin_file_path, npy_dir_path, cann_path):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        cann_path: user or system cann_path for using msaccucmp.py
    """
    python_version = sys.executable.split('/')[-1]
    msaccucmp_command_file_path = os.path.join(cann_path, MSACCUCMP_FILE_PATH)
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    utils.logger.info("convert dump data: %s to npy file" % (bin_file_path))
    utils.execute_command(bin2npy_cmd)