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

import time

import pandas as pd

from app_analyze.utils.log_util import logger
from app_analyze.scan.scanner import Scanner
from app_analyze.scan.clang_parser import Parser


class CxxScanner(Scanner):
    def __init__(self, files):
        super().__init__(files)

    def do_scan(self):
        start_time = time.time()
        result = self.exec_without_threads()
        self.porting_results['cxx'] = result
        eval_time = time.time() - start_time

        logger.info(f'Total time for scanning cxx files is {eval_time}s')

    def exec_without_threads(self):
        result = {}
        for file in self.files:
            p = Parser(file)
            rst_vals = p.parse()
            result[file] = pd.DataFrame.from_dict(rst_vals)

        return result
