#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
import logging
import os
import re
import sys
import subprocess
import numpy as np
from .config import Const
from .file_desc import DumpDecodeFileDesc, FileDesc
from .parse_exception import ParseException

try:
    from rich.traceback import install
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rich_print
    from rich.columns import Columns
    install()
except ImportError as err:
    install = None
    Panel = None
    Table = None
    Columns = None
    rich_print = None
    print("[Warning] Failed to import rich.", err)
    print("[Warning] Some features may not be available. Please run 'pip install rich' to fix it.")


class Util:
    def __init__(self):
        self.ms_accu_cmp = None
        logging.basicConfig(
            level=Const.LOG_LEVEL,
            format="%(asctime)s (%(process)d) -[%(levelname)s]%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.log = logging.getLogger()
        self.python = sys.executable

    @staticmethod
    def print(content):
        rich_print(content)

    @staticmethod
    def path_strip(path):
        return path.strip("'").strip('"')

    @staticmethod
    def _gen_npu_dump_convert_file_info(name, match, dir_path):
        return DumpDecodeFileDesc(name, dir_path, int(match.groups()[-4]), op_name=match.group(2),
                                  op_type=match.group(1), task_id=int(match.group(3)), anchor_type=match.groups()[-3],
                                  anchor_idx=int(match.groups()[-2]))

    @staticmethod
    def _gen_numpy_file_info(name, math, dir_path):
        return FileDesc(name, dir_path)

    def execute_command(self, cmd):
        if not cmd:
            self.log.error("Commond is None")
            return -1
        self.log.debug("[RUN CMD]: %s", cmd)
        cmd = cmd.split(" ")
        complete_process = subprocess.run(cmd, shell=False)
        return complete_process.returncode

    def print_panel(self, content, title='', fit=True):
        if not Panel:
            print(content)
            return
        if fit:
            self.print(Panel.fit(content, title=title))
        else:
            self.print(Panel(content, title=title))

    def check_msaccucmp(self, target_file):
        self.log.info("Try to auto detect file with name: %s.", target_file)
        result = subprocess.run(
            [self.python, target_file, "--help"], stdout=subprocess.PIPE)
        if result.returncode == 0:
            self.log.info("Check [%s] success.", target_file)
        else:
            self.log.error("Check msaccucmp failed in dir %s" % target_file)
            self.log.error("Please specify a valid msaccucmp.py path or install the cann package")
            raise ParseException(ParseException.PARSE_MSACCUCMP_ERROR)
        return target_file

    def create_dir(self, path):
        path = self.path_strip(path)
        if os.path.exists(path):
            return
        try:
            os.makedirs(path, mode=0o750)
        except OSError as e:
            self.log.error("Failed to create %s. %s", path, str(e))
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)

    def gen_npy_info_txt(self, source_data):
        shape, dtype, max_data, min_data, mean = \
            self.npy_info(source_data)
        return \
            '[Shape: %s] [Dtype: %s] [Max: %s] [Min: %s] [Mean: %s]' % (shape, dtype, max_data, min_data, mean)

    def save_npy_to_txt(self, data, dst_file='', align=0):
        if os.path.exists(dst_file):
            self.log.info("Dst file %s exists, will not save new one.", dst_file)
            return
        shape = data.shape
        data = data.flatten()
        if align == 0:
            align = 1 if len(shape) == 0 else shape[-1]
        elif data.size % align != 0:
            pad_array = np.zeros((align - data.size % align,))
            data = np.append(data, pad_array)
        np.savetxt(dst_file, data.reshape((-1, align)), delimiter=' ', fmt='%g')

    def list_convert_files(self, path, external_pattern=""):
        return self._list_file_with_pattern(
            path, Const.OFFLINE_DUMP_CONVERT_PATTERN, external_pattern, self._gen_npu_dump_convert_file_info
        )

    def list_numpy_files(self, path, extern_pattern=''):
        return self._list_file_with_pattern(path, Const.NUMPY_PATTERN, extern_pattern,
                                            self._gen_numpy_file_info)

    def create_columns(self, content):
        if not Columns:
            self.log.error("No Module named rich, please install it")
            raise ParseException(ParseException.PARSE_NO_MODULE_ERROR)
        return Columns(content)

    def create_table(self, title, columns):
        if not Table:
            self.log.error("No Module named rich, please install it and restart parse tool")
            raise ParseException(ParseException.PARSE_NO_MODULE_ERROR)
        table = Table(title=title)
        for column_name in columns:
            table.add_column(column_name, overflow='fold')
        return table

    def check_path_valid(self, path):
        path = self.path_strip(path)
        if not path or not os.path.exists(path):
            self.log.error("The path %s does not exist." % path)
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
        if os.path.islink(path):
            self.log.error('The file path {} is a soft link.'.format(path))
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
        if len(os.path.realpath(path)) > Const.DIRECTORY_LENGTH or len(os.path.basename(path)) > \
                Const.FILE_NAME_LENGTH:
            self.log.error('The file path length exceeds limit.')
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
        if not re.match(Const.FILE_PATTERN, os.path.realpath(path)):
            self.log.error('The file path {} contains special characters.'.format(path))
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)

        if os.path.isdir(path) and len(os.listdir(path)) == 0:
            self.log.error("No files in %s." % path)
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)

        if os.path.isfile(path):
            file_size = os.path.getsize(path)
            if path.endswith(Const.PKL_SUFFIX) and file_size > Const.ONE_GB:
                self.log.error('The file {} size is greater than 1GB.'.format(path))
                raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
            if path.endswith(Const.NPY_SUFFIX) and file_size > Const.TEN_GB:
                self.log.error('The file {} size is greater than 10GB.'.format(path))
                raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)

    def npy_info(self, source_data):
        if isinstance(source_data, np.ndarray):
            data = source_data
        else:
            self.log.error("Invalid data, data is not ndarray")
            raise ParseException(ParseException.PARSE_INVALID_DATA_ERROR)
        if data.dtype == 'object':
            self.log.error("Invalid data, data is object.")
            raise ParseException(ParseException.PARSE_INVALID_DATA_ERROR)
        if np.size(data) == 0:
            self.log.error("Invalid data, data is empty")
            raise ParseException(ParseException.PARSE_INVALID_DATA_ERROR)
        return data.shape, data.dtype, data.max(), data.min(), data.mean()

    def _list_file_with_pattern(self, path, pattern, extern_pattern, gen_info_func):
        self.check_path_valid(path)
        file_list = {}
        re_pattern = re.compile(pattern)
        for dir_path, dir_names, file_names in os.walk(path, followlinks=True):
            for name in file_names:
                match = re_pattern.match(name)
                if not match:
                    continue
                if extern_pattern != '' and not re.match(extern_pattern, name):
                    continue
                file_list[name] = gen_info_func(name, match, dir_path)
        return file_list

    def check_path_format(self, path, suffix):
        if os.path.isfile(path):
            if not path.endswith(suffix):
                self.log.error("%s is not a %s file." % (path, suffix))
                raise ParseException(ParseException.PARSE_INVALID_FILE_FORMAT_ERROR)
        elif os.path.isdir(path):
            self.log.error("Please specify a single file path")
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
        else:
            self.log.error("The file path %s is invalid" % path)
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
