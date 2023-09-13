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
import json
import numpy as np

from .config import Const
from .utils import Util
from .parse_exception import ParseException


class Visualization:
    def __init__(self):
        self.util = Util()

    def print_npy_summary(self, target_file):
        try:
            np_data = np.load(target_file, allow_pickle=True)
        except UnicodeError as e:
            self.util.log.error("%s %s" % ("UnicodeError", str(e)))
            self.util.log.warning("Please check the npy file")
            raise ParseException(ParseException.PARSE_UNICODE_ERROR)
        table = self.util.create_table('', ['Index', 'Data'])
        flatten_data = np_data.flatten()
        for i in range(min(16, int(np.ceil(flatten_data.size / 8)))):
            last_idx = min(flatten_data.size, i * 8 + 8)
            table.add_row(str(i * 8), ' '.join(flatten_data[i * 8: last_idx].astype('str').tolist()))
        summary = ['[yellow]%s[/yellow]' % self.util.gen_npy_info_txt(np_data), 'Path: %s' % target_file,
                   "TextFile: %s.txt" % target_file]
        self.util.print_panel(self.util.create_columns([table, "\n".join(summary)]), target_file)
        self.util.save_npy_to_txt(np_data, target_file + "txt")

    def print_npy_data(self, file_name):
        file_name = self.util.path_strip(file_name)
        self.util.check_path_valid(file_name)
        self.util.check_path_format(file_name, Const.NPY_SUFFIX)
        return self.print_npy_summary(file_name)

    def parse_pkl(self, path, api_name):
        path = self.util.path_strip(path)
        self.util.check_path_valid(path)
        self.util.check_path_format(path, Const.PKL_SUFFIX)
        with open(path, "r") as pkl_handle:
            title_printed = False
            while True:
                pkl_line = pkl_handle.readline()
                if pkl_line == '\n':
                    continue
                if len(pkl_line) == 0:
                    break
                try:
                    msg = json.loads(pkl_line)
                except json.JSONDecodeError as e:
                    self.util.log.error("%s %s in line %s" % ("JSONDecodeError", str(e), pkl_line))
                    self.util.log.warning("Please check the pkl file")
                    raise ParseException(ParseException.PARSE_JSONDECODE_ERROR)
                info_prefix = msg[0]
                if not info_prefix.startswith(api_name):
                    continue
                if info_prefix.find("stack_info") != -1 and len(msg) == 2:
                    print("\nTrace back({}):".format(msg[0]))
                    if msg[1] and len(msg[1]) > 4:
                        for item in reversed(msg[1]):
                            print("  File \"{}\", line {}, in {}".format(item[0], item[1], item[2]))
                            print("    {}".format(item[3]))
                        continue
                if len(msg) > 5:
                    summery_info = "  [{}][dtype: {}][shape: {}][max: {}][min: {}][mean: {}]" \
                        .format(msg[0], msg[3], msg[4], msg[5][0], msg[5][1], msg[5][2])
                    if not title_printed:
                        print("\nStatistic Info:")
                        title_printed = True
                    print(summery_info)
            pkl_handle.close()
