from report.report import Report
from utils.excel import write_excel
from utils.log_util import logger
from common.kit_config import KitConfig


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

    def initialize(self, project):
        self.report_content = project.get_results()

    def generate(self):
        if self.report_path == '':
            self.report_path = KitConfig.source_directory + '/' + 'output.xlsx'

        write_excel(self.report_content, self.report_path)

    def generate_abnormal(self, message):
        logger.info(message)
