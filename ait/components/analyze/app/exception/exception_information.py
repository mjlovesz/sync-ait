from enum import Enum, unique


@unique
class SourceScanErrorCode(Enum):
    """
    定义底层源码扫描与服务端代码的错误码
    """
    cmake_execute_failed = '2001'
    makefile_execute_failed = '2002'
    automake_execute_failed = '2003'
    source_scan_no_result = '2004'
    source_file_not_found = '2005'


@unique
class SourceScanErrorInfo(Enum):
    """
    源码扫描错误信息
    """
    source_scan_error_info = {
        '2001': {
            'cn': 'Cmake执行失败：',
            'en': 'Cmake execute failed: '
        },
        '2002': {
            'cn': 'Makefile执行失败：',
            'en': 'Makefile execute failed: '
        },
        '2003': {
            'cn': 'Automake执行失败：',
            'en': 'Automake execute failed: '
        },
        '2004': {
            'cn': '源码扫描没有结果',
            'en': 'Source scan with no results.'
        },
        '2005': {
            'cn': '源码文件丢失',
            'en': 'Source code not found.'
        }
    }

    @staticmethod
    def get_en_info(error_code):
        """ 获取源码扫描错误信息 """
        return SourceScanErrorInfo.source_scan_error_info.get(
            error_code).get('en')

    @staticmethod
    def get_cn_info(error_code):
        """ 获取源码扫描错误信息 """
        return SourceScanErrorInfo.source_scan_error_info.get(
            error_code).get('en')
