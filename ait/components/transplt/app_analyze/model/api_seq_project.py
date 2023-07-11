import time

from app_analyze.common.kit_config import KitConfig
from app_analyze.model.project import Project
from app_analyze.scan.scanner_factory import ScannerFactory
from app_analyze.scan.func_parser import FuncParser
from app_analyze.scan.sequence.seq_handler import SeqHandler


class APISeqProject(Project):
    def __init__(self, inputs):
        super().__init__(inputs)
        self.api_seqs = []

    def setup_scanners(self):
        """
        根据传入的扫描类型生成对应的扫描器实例，第一阶段只支持C/C++文件和
        makefile文件的扫描。
        说明：第一阶段这里的参数传递暂时做成这个样子，方便随时增减参数内容。
        但是问题是被调用方知道参数的内容才可以顺利取出。所以没做被调用方的取
        值失败的异常情况。要特别小心。后续可以考虑将参数包装成类进行传递。
        :return: NA
        """
        scanner_params = {
            'cpp_files': {
                'cpp': self.file_matrix.files.get('cpp_sources'),
                'hpp': self.file_matrix.files.get('hpp_sources'),
                'include_path': self.file_matrix.files.get('include_path'),
                'cxx_parser': FuncParser
            },
        }

        scanner_factory = ScannerFactory(scanner_params)
        for s_type in self.inputs.scanner_type:
            self.scanners.append(scanner_factory.get_scanner(s_type))

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

        val_dict = self.scan_results.get('cxx', None)
        if not val_dict:
            return

        # handle results
        rst = list(val_dict.values())[0]
        if len(val_dict) > 1:
            SeqHandler.union_api_seqs(rst)
        self.api_seqs = SeqHandler.clean_api_seqs(rst)

        eval_time = time.time() - start_time
        KitConfig.PROJECT_TIME = eval_time

    def get_api_seqs(self):
        return self.api_seqs
