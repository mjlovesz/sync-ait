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

import os
import numpy as np
from .utils import Util
from .config import Const
from .parse_exception import ParseException


class Compare:
    def __init__(self):
        self.util = Util()
        self.log = self.util.log
        self.vector_compare_result = {}
        self.msaccucmp = None

    @property
    def call_msaccucmp(self):
        if not self.msaccucmp:
            self.msaccucmp = self.util.check_msaccucmp(Const.MS_ACCU_CMP_PATH)
        return self.msaccucmp

    def npu_vs_npu_compare(self, my_dump_path, golden_dump_path, result_dir):
        self.log.info("Start Compare ...............")
        self.compare_vector(my_dump_path, golden_dump_path, result_dir)
        self.log.info("Compare finished!!")

    def compare_vector(self, my_dump_path, golden_dump_path, result_dir):
        self.util.create_dir(result_dir)
        cmd = '%s %s compare -m %s -g %s -out %s' % (
            self.util.python, self.call_msaccucmp, my_dump_path, golden_dump_path, result_dir
        )
        return self.util.execute_command(cmd)

    def convert_dump_to_npy(self, dump_file, data_format, output):
        file_name = ""
        if os.path.isfile(dump_file):
            self.log.info("Covert file is: %s", dump_file)
            file_name = os.path.basename(dump_file)
        elif os.path.isdir(dump_file):
            self.log.info("Convert all files in path: %s", dump_file)
            file_name = ""
        output = output if output else Const.DUMP_CONVERT_DIR
        self.util.check_path_valid(output)
        convert = self.convert(dump_file, data_format, output)
        if convert == 0:
            convert_files = self.util.list_convert_files(output, file_name)

            summary_txt = ["SrcFile: %s" % dump_file]
            for convert_file in convert_files.values():
                summary_txt.append(" - %s" % convert_file.file_name)
            self.util.print_panel("\n".join(summary_txt))

    def convert(self, dump_file, data_format, output):
        self.util.create_dir(output)
        if data_format:
            cmd = '%s %s convert -d %s -out %s -f %s' % (
                self.util.python, self.call_msaccucmp, dump_file, output, data_format
            )
        else:
            cmd = '%s %s convert -d %s -out %s' % (
                self.util.python, self.call_msaccucmp, dump_file, output
            )
        return self.util.execute_command(cmd)

    def compare_data(self, left, right, save_txt=False, rl=0.001, al=0.001, diff_count=20):
        """Compare data"""
        if left is None or right is None:
            raise ParseException("invalid input or output")
        try:
            left_data = np.load(left)
            right_data = np.load(right)
        except UnicodeError as e:
            self.log.error("%s %s" % ("UnicodeError", str(e)))
            self.log.warning("Please check the npy file")
            raise ParseException(ParseException.PARSE_UNICODE_ERROR)
        except IOError:
            self.log.error("Failed to load npy %s or %s." % (left, right))
            raise ParseException(ParseException.PARSE_LOAD_NPY_ERROR)

        # save to txt
        if save_txt:
            self.util.save_npy_to_txt(left_data, left + ".txt")
            self.util.save_npy_to_txt(right_data, right + ".txt")
        # compare data
        total_cnt, all_close, cos_sim, err_percent = self._do_compare_data(left_data, right_data, rl, al, diff_count)
        content = ['Left:', ' ├─ NpyFile: %s' % left]
        if save_txt:
            content.append(' ├─ TxtFile: [green]%s.txt[/green]' % left)
        content.append(' └─ NpySpec: [yellow]%s[/yellow]' % self.util.gen_npy_info_txt(left_data))
        content.append('Right:')
        content.append(' ├─ NpyFile: %s' % right)
        if save_txt:
            content.append(' ├─ TxtFile: [green]%s.txt[/green]' % right)
        content.append(' └─ NpySpec: [yellow]%s[/yellow]' % self.util.gen_npy_info_txt(right_data))
        content.append('NumCnt:   %s' % total_cnt)
        content.append('AllClose: %s' % all_close)
        content.append('CosSim:   %s' % cos_sim)
        content.append('ErrorPer: %s  (rl= %s, al= %s)' % (err_percent, rl, al))
        self.util.print_panel("\n".join(content))

    def _do_compare_data(self, left, right, rl=0.001, al=0.001, diff_count=20):
        data_left = left.astype(np.float32)
        data_right = right.astype(np.float32)
        shape_left = data_left.shape
        shape_right = data_right.shape
        if shape_left != shape_right:
            self.log.warning("Data shape not equal: %s vs %s", data_left.shape, data_right.shape)
        data_left = data_left.reshape(-1)
        data_right = data_right.reshape(-1)
        if data_left.shape[0] != data_right.shape[0]:
            self.log.warning("Data size not equal: %s vs %s", data_left.shape, data_right.shape)
            if data_left.shape[0] < data_right.shape[0]:
                data_left = np.pad(data_left, (0, data_right.shape[0] - data_left.shape[0]), 'constant')
            else:
                data_right = np.pad(data_right, (0, data_left.shape[0] - data_right.shape[0]), 'constant')
        all_close = np.allclose(data_left, data_right, atol=al, rtol=rl)
        cos_sim = np.dot(data_left, data_right) / (
                np.sqrt(np.dot(data_left, data_left)) * np.sqrt(np.dot(data_right, data_right)))
        err_cnt = 0
        total_cnt = data_left.shape[0]
        diff_table_columns = ['Index', 'Left', 'Right', 'Diff']
        err_table = self.util.create_table("Error Item Table", diff_table_columns)
        top_table = self.util.create_table("Top Item Table", diff_table_columns)
        for i in range(total_cnt):
            abs_diff = abs(data_left[i] - data_right[i])
            if i < diff_count:
                top_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
            if abs_diff > (al + rl * abs(data_right[i])):
                if err_cnt < diff_count:
                    err_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
                err_cnt += 1
        err_percent = float(err_cnt / total_cnt)
        self.util.print(self.util.create_columns([err_table, top_table]))
        return total_cnt, all_close, cos_sim, err_percent
