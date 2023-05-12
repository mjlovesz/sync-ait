# Copyright 2023 Huawei Technologies Co., Ltd
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
from pathlib import Path

from porting.porting_input import IInput
from common.kit_config import KitConfig, ReporterType
from utils.io_util import IOUtil


class CommandLineInput(IInput):
    """
    CommandLineInput对象表示用户的输入来自命令行
    """

    # 继承父类的 __slots__
    __slots__ = []

    def __init__(self, args=None):
        super().__init__(args)

    @staticmethod
    def _check_path(folder):
        if not os.path.exists(folder):
            raise ValueError("{} porting-advisor: error: {}".
                             format(KitConfig.porting_content,
                                    'The path %s does not exist or you do not '
                                    'have the permission to access the path. '
                                    % folder))
        elif not os.path.isdir(folder):
            raise ValueError("{} porting-advisor: error: {}".
                             format(KitConfig.porting_content,
                                    'The path %s is '
                                    'not directory. ' % folder))
        elif not os.access(folder, os.R_OK):
            raise ValueError("{} porting-advisor: error: {}".
                             format(KitConfig.porting_content,
                                    "Cannot access the file "
                                    "or directory: %s" % folder))
        elif Path(folder).is_dir() and not os.access(folder, os.X_OK):
            raise ValueError("{} porting-advisor: error: {}".
                             format(KitConfig.porting_content,
                                    "Cannot access the "
                                    "directory: %s" % folder))
        elif IOUtil.check_path_is_empty(folder):
            raise ValueError("{} porting-advisor: error: {}".
                             format(KitConfig.porting_content,
                                    'The directory %s '
                                    'is empty' % folder))

    def resolve_user_input(self):
        """解析来自命令行的用户输入"""
        self._get_source_directories()
        self._get_construct_tool()
        self._get_debug_switch()
        self._get_output_type()
        self.set_scanner_type()

    def _get_source_directories(self):
        if not self.args.source:
            raise ValueError('porting-advisor: error: '
                             'the following arguments are '
                             'required: s/--source')
        if self.args.source:
            for folder in self.args.source.split(','):
                folder = folder.strip()
                folder.replace('\\', '/')
                folder = os.path.realpath(folder)
                self._check_path(folder)
                if not folder.endswith('/'):
                    folder += '/'
                self.directories.append(folder)
            self.directories = sorted(set(self.directories),
                                      key=self.directories.index)
            self.source_path = self.directories
            self.directories = \
                IOUtil.remove_subdirectory(self.directories)

    def _get_construct_tool(self):
        """获取构建工具类型"""
        if not self.args.tools:
            self.args.tools = 'make'
        if self.args.tools not in KitConfig.valid_construct_tools:
            raise ValueError('{} porting-advisor: error: construct '
                             'tool {} is not supported. supported '
                             'input are '
                             '{}.'.format(KitConfig.porting_content,
                                          self.args.tools,
                                          ' or '.join(KitConfig.
                                                      valid_construct_tools)))
        self.construct_tool = self.args.tools

    def _get_debug_switch(self):
        """动态修改日志级别"""
        self.debug_switch = self.args.log_level

    def _get_output_type(self):
        """获取输出报告格式"""
        out_format = self.args.report_type.lower()
        if out_format not in KitConfig.valid_report_type:
            raise ValueError('porting-advisor: error: output type {} is not '
                             'supported. supported input '
                             'is csv/JSON.'.format(self.args.report_type))

        if out_format == 'csv':
            self.report_type.append(ReporterType.CSV_REPORTER)
        if out_format == 'json':
            self.report_type.append(ReporterType.JSON_REPORTER)
