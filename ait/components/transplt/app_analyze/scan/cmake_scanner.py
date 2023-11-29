# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import re
import os
import time
from collections import namedtuple, OrderedDict

import pandas as pd
import numpy as np

from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.scanner import Scanner
from app_analyze.scan import scanner_utils
from app_analyze.scan.module.comment_delete import CommentDelete
from app_analyze.utils.log_util import logger


class CMakeScanner(Scanner):
    """
    cmake扫描器的具体子类
    """
    PATTERN = r'(?:(?P<data>(?P<key_word>(.*?))(?P<data_inner>((?:\s*\()([^\)]+)))\)))'
    SAVE_VAR_INFO_INPUT = namedtuple('save_var_info_input',
                                     ['func_name', 'body', 'start_line', 'match_flag', 'var_def_dict'])

    def __init__(self, files):
        super().__init__(files)
        self.name = 'CMakelists.txt'
        self.var_rel_commands = ['set', 'find_file', 'find_library', 'find_path', 'aux_source_directory',
                                 'pkg_check_modules']
        self.marco_pattern = r'\$\{(.*?)\}'
        self.pkg_pattern = r'PkgConfig::([0-9a-zA-Z]+)'

    @staticmethod
    def _check_var_info(val, start_line, var_def_dict):
        locs = var_def_dict[val]
        lines = list(locs.keys())
        idx = np.searchsorted(lines, start_line)
        flag = locs[lines[idx - 1]]
        return flag

    @staticmethod
    def _read_cmake_file_content(filepath):
        """
        功能：读取CMakelists.txt文件内容，并删除注释
        :param filepath:文件路径
        :return:去掉注释后的文件内容
        """
        with open(filepath, errors='ignore') as file_desc:
            try:
                contents = file_desc.read()
            except UnicodeDecodeError as err:
                logger.error('%s decode error. Only the utf-8 format is '
                             'supported. Except:%s.', filepath, err)
                contents = ""
        contents = CommentDelete(contents, '#', CommentDelete.MULTI_COMMENT_CMAKE).delete_comment()
        return contents

    def do_scan(self):
        start_time = time.time()
        result = self._do_cmake_scan_with_file()
        self.porting_results['cmake'] = result
        self.update_include_dirs()
        eval_time = time.time() - start_time

        if result:
            logger.info(f'Total time for scanning cmake files is {eval_time}s')


    def update_include_dirs(self):
        """
        add_subdirectory (source_dir [binary_dir] [EXCLUDE_FROM_ALL])
        include_directories ([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
        """
        pattern = r'include_directories\(([^)]+)\)'
        result = []
        if self.files:
            # 获取工程根目录，后续作为路径前缀，默认CMakeLists.txt[0]在项目根路径下
            root_dir = os.path.dirname(self.files[0])
            for file in self.files:
                # 对外部传入路径进行校验
                if not os.path.isfile(file):
                    continue
                # 获取目录名，后续作为include_directories的路径前缀
                with open(file) as f:
                    cmake_content = f.read()
                include_directories_pattern = re.compile(pattern, re.DOTALL)
                matches = include_directories_pattern.findall(cmake_content)
                if matches:
                    result_path = self.match_paths(matches, root_dir)
                    result.extend(result_path)
            else:
                logger.info(f'No include_directories found in the CMakeLists.txt file.')

            # 修改KitConfig.INCLUDES并去重
        result = [x for x in result if result.count(x) == 1]
        for i, item in enumerate(result):
            key = "CMakeLists_{i}".format(i=i)
            KitConfig.INCLUDES[key] = item

    def match_paths(self, matches, root_dir):
        result = []
        for match in matches:
            paths = [path.strip('\"').strip() for path in match.split('\n') if path.strip()]
            for path in paths:
                # 根目录下的路径
                processed_root_path = os.path.join(root_dir, path)
                if os.path.exists(processed_root_path):
                    result.append(processed_root_path)
                # 处理变量引用，如${PROJECT_SOURCE_DIR}和${OpenCV_INCLUDE_DIRS}
                processed_path = re.sub(r'\$\{([^}]+_SOURCE_DIR)\}/([^;"\)]+)',
                                        lambda match: os.path.join(root_dir, match.group(2)), path)
                # 处理直接路径
                if processed_path.startswith(('AFTER ', 'BEFORE ', 'SYSTEM ')):
                    processed_path = processed_path.split(' ', 1)[1].strip()
                if not os.path.exists(processed_path):
                    continue
                result.append(processed_path)
        return result


    def _do_cmake_scan_with_file(self):
        """
        功能：cmake全量扫描
        :return:
        """
        # 全量扫描对cmake添加编译选项只需对根路径的CMakelist.txt处理
        result = {}
        for file in self.files:
            # 对外部传入路径进行校验
            if not os.path.isfile(file):
                continue
            logger.info(f"Scanning file: {file}.")
            rst_vals = self._scan_cmake_function(file)
            result[file] = pd.DataFrame.from_dict(rst_vals)

        return result

    def _scan_cmake_function(self, filepath):
        rst_dict = {}
        var_def_dict = OrderedDict()

        contents = self._read_cmake_file_content(filepath)
        match = re.finditer(CMakeScanner.PATTERN, contents, re.M)

        for item in match:
            start_line, end_line = scanner_utils.get_line_number(contents, item)

            content = item['data']
            func_name = item['key_word']

            body = item['data_inner']
            match_flag = False
            if KitConfig.MACRO_PATTERN.search(content) or KitConfig.LIBRARY_PATTERN.search(
                    content) or KitConfig.FILE_PATTERN.search(content):
                # exact match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst

                match_flag = True
            elif self._check_var_ref_info(body, start_line, var_def_dict):
                # reference match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst
            elif KitConfig.KEYWORD_PATTERN.search(content):
                # fuzzy match
                rst = {'lineno': start_line, 'content': content, 'command': func_name, 'suggestion': 'modifying'}
                rst_dict[start_line] = rst
            # save variable definition
            save_var_info_input = CMakeScanner.SAVE_VAR_INFO_INPUT(
                func_name, body, start_line, match_flag, var_def_dict)
            self._save_var_info(save_var_info_input)

        return list(rst_dict.values())

    def _check_var_ref_info(self, body, start_line, var_def_dict):
        macros = []
        vals = re.findall(self.marco_pattern, body)
        macros.extend(vals)

        vals = re.findall(self.pkg_pattern, body)
        macros.extend(vals)

        for macro in macros:
            if var_def_dict.get(macro) is None:
                continue

            if self._check_var_info(macro, start_line, var_def_dict):
                return True

        return False

    def _save_var_info(self, save_var_info_input):
        func_name, body, start_line, match_flag, var_def_dict = save_var_info_input
        if func_name in self.var_rel_commands:
            # var define
            words = body.replace('(', '').strip().split(' ')
            if func_name == 'aux_source_directory':
                var_name = words[-1]
            else:
                var_name = words[0]

            if var_def_dict.get(var_name) is None:
                var_def_dict[var_name] = {start_line: match_flag}
            else:
                var_def_dict[var_name][start_line] = match_flag
