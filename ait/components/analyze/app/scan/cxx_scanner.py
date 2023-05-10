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

import time

import pandas as pd

from utils.thread_helper import alloc_configs_for_subprocess, MyThread
from utils.log_util import logger
from common.kit_config import KitConfig
from scan.scanner import Scanner
from scan.clang_parser import Parser


class CxxScanner(Scanner):
    def __init__(self, files):
        super().__init__(files)

    @staticmethod
    def eval_thread(files, rp_idx, tid):
        result = {}
        for idx in range(rp_idx[tid], rp_idx[tid + 1]):
            cxx_f = files[idx]

            p = Parser(cxx_f)
            rst_vals = p.parse(log=KitConfig.print_detail)

            result[cxx_f] = pd.DataFrame.from_dict(rst_vals)
        return result

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
            rst_vals = p.parse(log=KitConfig.print_detail)
            result[file] = pd.DataFrame.from_dict(rst_vals)

        return result

    def exec_with_threads(self):
        rp_idx = alloc_configs_for_subprocess(KitConfig.thread_num, len(self.files))
        threads = []
        for tid in range(KitConfig.thread_num):
            thread = MyThread(func=self.eval_thread, args=(self.files, rp_idx, tid))
            threads.append(thread)
            thread.start()

        result = {}
        for thread in threads:
            thread.join()
            obj_apis = thread.get_result()
            result.update(obj_apis)

        return result
