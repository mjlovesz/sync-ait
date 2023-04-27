from abc import ABC, abstractmethod
from common.kit_config import ReporterType, ScannerType


class IInput(ABC):
    """
    抽象输入类型接口
    """
    __slots__ = [
        'args',  # 保存接收的输入参数原始类型对象
        'source_path',  # 源码路径
        'directories',  # 工具需要进行扫描的文件夹路径
        'report_type',  # 扫描生成的迁移报告的类型
        'scanner_type',  # 扫描器种类，第一阶段是默认的
        'construct_tool',  # 指定的构建工具，可以是cmake或者make（默认）
        'project_directory',  # 本次扫描任务的输出目录
        'project_time',  # 本次扫描任务的时间记录字符串
        'worker_temp_dir',  # worker的临时目录
        'workspace_path',  # 用户工作空间
        'debug_switch',  # 日志级别
    ]

    def __init__(self, args=None):
        self.args = args
        self.source_path = []
        self.directories = []
        self.report_type = []
        self.scanner_type = []
        self.construct_tool = 'cmake'
        self.project_directory = ''
        self.project_time = ''
        self.worker_temp_dir = ''
        self.debug_switch = 'INFO'

    @abstractmethod
    def resolve_user_input(self):
        """解析用户输入"""
        raise NotImplementedError(
            r'abstract interface. need subclass to implementation.')

    def set_scanner_type(self):
        if self.construct_tool == "cmake":
            self.scanner_type.append(ScannerType.CMAKE_SCANNER)
            self.scanner_type.append(ScannerType.CPP_SCANNER)
        else:
            NotImplementedError('need to implementation.')
