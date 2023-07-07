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

import os

from app_analyze.porting.cmdline_input import CommandLineInput
from app_analyze.common.kit_config import KitConfig, ReporterType
from app_analyze.utils.io_util import IOUtil


class CustomInput(CommandLineInput):
    """
    CommandLineInput对象表示用户的输入来自命令行
    """

    # 继承父类的 __slots__
    __slots__ = []

    def __init__(self, args=None):
        super().__init__(args)

    def get_source_directories(self):
        if not self.args.source:
            raise ValueError('No input files!')
        else:
            for folder in self.args.source.split(','):
                folder = folder.strip()
                folder.replace('\\', '/')
                folder = os.path.realpath(folder)
                if os.path.isdir(folder):
                    self._check_path(folder)
                    if not folder.endswith('/'):
                        folder += '/'

                self.directories.append(folder)
            self.directories = sorted(set(self.directories),
                                      key=self.directories.index)
            self.source_path = self.directories
            self.directories = \
                IOUtil.remove_subdirectory(self.directories)
