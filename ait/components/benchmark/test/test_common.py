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

import filecmp
import math
import os
import random
import shutil
import sys
from fileinput import filename

import aclruntime
import numpy as np


class TestCommonClass:
    default_device_id = 0
    EPSILON = 1e-6
    epsilon = 1e-6
    cmd_prefix = (
        sys.executable + " " + os.path.join(os.path.dirname(os.path.realpath(__file__)), "../ais_bench/__main__.py")
    )
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../test/testdata")
    msame_bin_path = os.getenv('MSAME_BIN_PATH')

    @staticmethod
    def get_cmd_prefix():
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return sys.executable + " " + os.path.join(_current_dir, "../ais_bench/__main__.py")

    @staticmethod
    def get_basepath():
        """
        test/testdata
        """
        data_path = os.getenv("AIT_BENCHMARK_DT_DATA_PATH")
        if not data_path:
            _current_dir = os.path.dirname(os.path.realpath(__file__))
            return os.path.join(_current_dir, "../test/testdata")
        else:
            return os.path.realpath(data_path)

    @staticmethod
    def create_inputs_file(input_path, size, pure_data_type=random):
        file_path = os.path.join(input_path, "{}.bin".format(size))
        if pure_data_type == "zero":
            lst = [0 for _ in range(size)]
        else:
            lst = [random.randrange(0, 256) for _ in range(size)]
        barray = bytearray(lst)
        ndata = np.frombuffer(barray, dtype=np.uint8)
        ndata.tofile(file_path)
        return file_path

    @staticmethod
    def prepare_dir(target_folder_path):
        if os.path.exists(target_folder_path):
            shutil.rmtree(target_folder_path)
        os.makedirs(target_folder_path, 0o750)

    @staticmethod
    def get_model_inputs_size(model_path):
        options = aclruntime.session_options()
        session = aclruntime.InferenceSession(model_path, TestCommonClass.default_device_id, options)
        return [meta.realsize for meta in session.get_inputs()]

    @staticmethod
    def get_inference_execute_num(log_path):
        if not os.path.exists(log_path) and not os.path.isfile(log_path):
            return 0

        cmd = "cat {} |grep 'model aclExec cost :' | wc -l".format(log_path)
        try:
            outval = os.popen(cmd).read()
        except Exception as e:
            raise Exception("grep action raises raise an exception: {}".format(e)) from e

        return int(outval.replace('\n', ''))

    @classmethod
    def get_inputs_path(cls, size, input_path, input_file_num, pure_data_type=random):
        """generate input files
        folder structure as follows.
        input_path
                |_ 196608           # size_path
                    |- 196608.bin   # base_size_file
                    |_ 5            # input_file_num_folder_path

        """
        size_path = os.path.join(input_path, str(size))
        if not os.path.exists(size_path):
            os.makedirs(size_path, 0o750)

        base_size_file_path = os.path.join(size_path, "{}.bin".format(size))
        if not os.path.exists(base_size_file_path):
            cls.create_inputs_file(size_path, size, pure_data_type)

        input_file_num_folder_path = os.path.join(size_path, str(input_file_num))

        if os.path.exists(input_file_num_folder_path):
            if len(os.listdir(input_file_num_folder_path)) == input_file_num:
                return input_file_num_folder_path
            else:
                shutil.rmtree(input_file_num_folder_path)

        if not os.path.exists(input_file_num_folder_path):
            os.makedirs(input_file_num_folder_path, 0o750)

        strs = []
        # create soft link to base_size_file
        for i in range(input_file_num):
            file_name = "{}-{}.bin".format(size, i)
            file_path = os.path.join(input_file_num_folder_path, file_name)
            strs.append("cp {} {}".format(base_size_file_path, file_path))
            strs.append(f"chmod 750 {file_path}")

        cmd = ';'.join(strs)
        os.system(cmd)

        return input_file_num_folder_path

    @classmethod
    def get_model_static_om_path(cls, batchsize, modelname):
        base_path = cls.get_basepath()
        return os.path.join(base_path, "{}/model".format(modelname), "pth_{}_bs{}.om".format(modelname, batchsize))
