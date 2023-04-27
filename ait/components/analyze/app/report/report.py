from common.kit_config import FileType


class Report:
    """
    Report类作为输出报告的抽象基类存在，仅定义必要的属性和方法接口。
    第一个版本只支持csv格式的输出，所以相关的操作
    都在csv_report模块中进行实现；后续增加新的输出格式时，
    需要增加新的输出格式模块，并在project模块中增加实例化
    具体子类的分支.
    """
    FILE_TYPE_KEY = {
        FileType.MAKEFILE: "make",
        FileType.CMAKE_LISTS: "cmake_lists",
    }
    C_LINES_KEY = ("c", "make", "cmake_lists",)

    def __init__(self, report_param):
        """Report实例初始化函数"""
        self.report_path = report_param['directory']
        self.report_time = report_param['project_time']

    def initialize(self, project):
        """抽象基类不是先该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement initialize method!'
                                  .format(self.__class__))

    def generate(self):
        """抽象基类不实现该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement generate method!'
                                  .format(self.__class__))

    def generate_abnormal(self, message):
        """抽象基类不实现该方法，交由各个子类实现"""
        raise NotImplementedError('{} must implement generate_abnormal method!'
                                  .format(self.__class__))
