from common.kit_config import ReporterType
from report.csv_report import CsvReport
from utils.log_util import logger


class ReporterFactory:
    """
    报告工厂类, 定义了输出文件后缀和具体处理类之间的对应关系
    """

    def __init__(self, report_params):
        """报告工厂类实例化函数"""
        self.report_params = report_params

    def get_reporter(self, report_type, info=None):
        """
        如果后面要增加新的输出报告格式，需要新增一个具体的报告子类，添加报告枚举，并在这里增加实例化的逻辑
        :param report_type: 报告枚举类型
        :param info: 任务基本信息
        :return: 报告实例对象
        """
        if report_type == ReporterType.CSV_REPORTER:
            return CsvReport(self.report_params)

        raise Exception('only support Csv report format.')

    def dump(self):
        """
        打印信息
        :return:NA
        """
        logger.debug(self.report_params)
