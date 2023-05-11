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
import os
import pandas as pd
from utils.log_util import logger
from common.kit_config import KitConfig
from report.report_factory import ReporterFactory
from scan.scanner_factory import ScannerFactory
from scan.module.file_matrix import FileMatrix
from solution.advisor import Advisor


class Project:
    """
    Project对象表示了迁移扫描任务相关属性和方法的集合
    """
    __slots__ = [
        'inputs',  # inputs对象实例
        'scanners',  # 预定义的扫描器类型列表：[Scanner]
        'reporters',  # 预定义的迁移报告类型列表： [Report]
        'file_matrix',  # 待扫描文件矩阵实例对象
        'scan_results',  # 汇总的扫描结果列表：PortingResult
        'lib_reports',  # 依赖字典处理之后的扫描结果
        'report_results',  # 推荐结果
    ]

    def __init__(self, inputs):
        self.inputs = inputs
        self.scanners = []
        self.reporters = []
        self.file_matrix = FileMatrix(self.inputs)
        self.scan_results = {}
        self.lib_reports = []
        self.report_results = {}

    def dump(self):
        """
        显示Project的内容
        :return:
        """
        logger.debug('------------------------------ project ------------------------------------')
        # 打印inputs
        logger.debug('[Scan Directories]: %s', self.inputs.directories)
        logger.debug('[ReportType]: %s', self.inputs.report_type)
        logger.debug('[ConstructTool]: %s', self.inputs.construct_tool)
        logger.debug('[ProjectTime]: %s', KitConfig.project_time)

    def setup_file_matrix(self):
        """
        根据传入的源文件目录先做一次文件的查找
        :return: 源代码文件列表和makefile文件列表以及汇编文件列表
        """
        self.file_matrix.setup_file_matrix()

    def setup_reporters(self, info):
        """
        根据传入的报告类型生成对应的报告实例
        说明：第一阶段这里的参数传递暂时做成这个样子，方便随时增减参数内容。
        但是问题是被调用方知道参数的内容才可以顺利取出。所以没做被调用方的取
        值失败的异常情况。要特别小心。后续可以考虑将参数包装成类进行传递。
        :return: NA
        """
        report_params = {
            'directory': self.inputs.project_directory,
            'project_time': self.inputs.project_time
        }

        report_factory = ReporterFactory(report_params)
        for r_type in self.inputs.report_type:
            self.reporters.append(report_factory.get_reporter(r_type, info))

    def setup_scanners(self):
        """
        根据传入的扫描类型生成对应的扫描器实例，第一阶段只支持C/C++文件和
        makefile文件的扫描。
        说明：第一阶段这里的参数传递暂时做成这个样子，方便随时增减参数内容。
        但是问题是被调用方知道参数的内容才可以顺利取出。所以没做被调用方的取
        值失败的异常情况。要特别小心。后续可以考虑将参数包装成类进行传递。
        :return: NA
        """
        makefiles = self.file_matrix.files.get('makefiles')
        cmake_files = self.file_matrix.files.get("cmakefiles")

        scanner_params = {
            'makefiles': makefiles,
            'cpp_files': {
                "cpp": self.file_matrix.files.get('cpp_sources'),
                "hpp": self.file_matrix.files.get('hpp_sources'),
                'include_path': self.file_matrix.files.get('include_path'),
            },
            'cmake_files': cmake_files,
        }

        scanner_factory = ScannerFactory(scanner_params)
        for s_type in self.inputs.scanner_type:
            self.scanners.append(scanner_factory.get_scanner(s_type))

    @staticmethod
    def _dedup(val_dict):
        rst_dict = {}
        for f, df in val_dict.items():
            vals = df.to_dict()
            if not vals:
                continue
            print()
            for idx, loc_str in vals['Location'].items():
                loc_info = loc_str.split(',')
                new_f = loc_info[0]
                loc = loc_info[1]

                if rst_dict.get(new_f) is None:
                    rst_dict[new_f] = {loc: {'API': vals['API'][idx],
                                             'CUDAEnable': vals['CUDAEnable'][idx],
                                             'Location': vals['Location'][idx],
                                             'Context(形参 | 实参 | 来源代码 | 来源位置)':
                                                 vals['Context(形参 | 实参 | 来源代码 | 来源位置)'][idx]}}
                else:
                    pre_val = rst_dict[new_f]
                    if pre_val.get(loc) is None:
                        rst_dict[new_f][loc] = {'API': vals['API'][idx],
                                                'CUDAEnable': vals['CUDAEnable'][idx],
                                                'Location': vals['Location'][idx],
                                                'Context(形参 | 实参 | 来源代码 | 来源位置)':
                                                    vals['Context(形参 | 实参 | 来源代码 | 来源位置)'][idx]}

        val_dict = {}
        for f, loc_dict in rst_dict.items():
            val_dict[f] = pd.DataFrame.from_dict(list(loc_dict.values()))

        return val_dict

    def scan(self):
        """
        调用定义的所有扫描器的scan函数进行扫描任务，核心并行扫描处理框架
        在这个函数里面
        :return: NA
        """
        if self.scanners is None:
            raise ValueError('Scanners is none')

        start_time = time.time()
        for scanner in self.scanners:
            scanner.do_scan()
            if scanner.porting_results is not None:
                self.scan_results.update(scanner.porting_results)

        for key, val_dict in self.scan_results.items():
            if key == 'cxx':
                if not val_dict:
                    continue

                rst_dict = self._dedup(val_dict)
                ad = Advisor(rst_dict, os.path.abspath(os.path.dirname(__file__)) + '/' + KitConfig.api_map)
                ad.recommend()
                workloads = ad.workload()
                logger.info(f'Workloads:\n', workloads)
                ad.cuda_apis()
                self.report_results.update(ad.results)
            elif key == 'cmake':
                self.report_results.update(val_dict)

        eval_time = time.time() - start_time
        KitConfig.project_time = eval_time

    def get_results(self):
        return self.report_results

    def generate_report(self, normal_report, message=None):
        """
        根据设置的报告类型实际生成报告
        :return: NA
        """
        for reporter in self.reporters:
            reporter.initialize(self)

        if normal_report:
            for report in self.reporters:
                report.generate()
        else:
            self.scan_results.clear()
            for report in self.reporters:
                report.generate_abnormal(message)
