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


def data_convert(npu_dump_data_path, npu_net_output_data_path, arguments):
    """
    Function Description:
        provide the interface for dump data conversion
    Exception Description:
        when invalid msaccucmp command throw exception
    """
    if _check_convert_bin2npy(arguments):
        common_path = os.path.commonprefix([npu_dump_data_path, npu_net_output_data_path])
        npu_dump_data_path_diff = os.path.relpath(npu_dump_data_path, common_path)
        time_stamp_file_path = npu_dump_data_path_diff.split(os.path.sep)[1]
        convert_dir_path = npu_dump_data_path.replace(time_stamp_file_path, time_stamp_file_path+'_bin2npy')
        convert_dir_path = os.path.normpath(convert_dir_path)
        convert_data_path = _check_data_convert_file(convert_dir_path)
        msaccucmp_command_file_path = os.path.join(arguments.cann_path, MSACCUCMP_FILE_PATH)
        python_version = sys.executable.split('/')[-1]
        bin2npy_cmd = [python_version, msaccucmp_command_file_path,
                        "convert", "-d", npu_dump_data_path, "-out", convert_data_path]
        utils.execute_command(bin2npy_cmd, True)
        utils.logger.info("msaccucmp command line: %s " % " ".join(bin2npy_cmd))
        return convert_data_path
    return ""


def data_convert_file(bin_file_path, npy_dir_path, arguments):
    """
    Function Description:
        convert a bin file to npy file.
    Parameter:
        bin_file_path: the path of the bin file needed to be converted to npy
        npy_dir_path: the dest dir to save the converted npy file
        arguments: the enter arguments, here arguments.cann_path is required
    """
    python_version = sys.executable.split('/')[-1]
    msaccucmp_command_file_path = os.path.join(arguments.cann_path, MSACCUCMP_FILE_PATH)
    bin2npy_cmd = [python_version, msaccucmp_command_file_path, "convert", "-d", bin_file_path, "-out", npy_dir_path]
    utils.logger.info("convert dump data: %s to npy file" % (bin_file_path))
    utils.execute_command(bin2npy_cmd, True)


def _check_data_convert_file(convert_dir_path):
    if not os.path.exists(convert_dir_path):
        os.makedirs(convert_dir_path)
    return convert_dir_path


def _check_convert_bin2npy(arguments):
    return arguments.bin2npy