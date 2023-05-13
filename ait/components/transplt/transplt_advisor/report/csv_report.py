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

from transplt_advisor.report.report import Report
from transplt_advisor.utils.excel import write_excel
from transplt_advisor.utils.log_util import logger
from transplt_advisor.common.kit_config import KitConfig


class CsvReport(Report):
    """
    CsvReport代表输出格式为csv的报告文件对象
    """

    def __init__(self, report_param):
        """实例化Csv报告对象"""
        super(CsvReport, self).__init__(report_param)
        self.report_content = {}

    def __repr__(self):
        """字符串表示"""
        return 'report_path: %s' % self.report_path

    @staticmethod
    def generate_abnormal(message):
        logger.info(message)

    def initialize(self, project):
        self.report_content = project.get_results()

    def generate(self):
        if self.report_path == '':
            self.report_path = KitConfig.source_directory + '/' + 'output.xlsx'

        write_excel(self.report_content, self.report_path)
        logger.info(f'Report generated at: {self.report_path}')
