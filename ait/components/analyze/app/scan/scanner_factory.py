from common.kit_config import ScannerType
from scan.cxx_scanner import CxxScanner
from scan.cmake_scanner import CMakeScanner


def merge_dicts(*dict_args):
    """ 字典合并 """
    result = {}
    for item in dict_args:
        result.update(item)

    return result


class ScannerFactory:
    """
    扫描器工厂类
    """

    def __init__(self, scanner_params):
        """实例化扫描器工厂对象"""
        self.scanner_params = scanner_params

    def get_scanner(self, scanner_type):
        """
        工厂生产方法
        :param scanner_type: 扫描器种类
        :return: 扫描器实例对象
        """
        if scanner_type == ScannerType.CPP_SCANNER:
            return CxxScanner(list(self.scanner_params['cpp_files']['cpp'].keys()))
        if scanner_type == ScannerType.CMAKE_SCANNER:
            return CMakeScanner(self.scanner_params['cmake_files'])
        raise Exception('Impossible Scanner Type!')
