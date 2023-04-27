from exception.exception_information import SourceScanErrorCode, SourceScanErrorInfo


class SourceScanException(Exception):
    """
    定义二进制异常类的基类，在scan_api入口进行异常捕获与抛出
    设置未知错误码为 9991
    """

    def __init__(self, error_info=''):
        super().__init__(error_info)
        self.error_code = '9991'
        self.error_info = error_info

    def __str__(self):
        return self.error_code + self.error_info


class CmakeExecuteFailedException(SourceScanException):
    """
    定义源码扫描Cmake执行失败异常
    """

    def __init__(self, error_info=''):
        super().__init__()
        self.error_code = SourceScanErrorCode.cmake_execute_failed.value
        self.error_info = error_info

    def __str__(self):
        return self.error_code + self.error_info


class MakefileExecuteFailException(SourceScanException):
    """
    定义源码扫描Makefile执行失败异常
    """

    def __init__(self, error_info='', error_key=None):
        super().__init__()
        self.error_code = SourceScanErrorCode.makefile_execute_failed.value
        self.error_info = error_info
        self.error_key = error_key

    def __str__(self):
        return self.error_code + self.error_info

    def get_error_info(self):
        return self.error_info


class AutomakeExecuteFailedException(SourceScanException):
    """
    定义源码扫描Cmake解析失败异常
    """

    def __init__(self, error_info='', am_err_key=None):
        super().__init__()
        self.error_code = SourceScanErrorCode.automake_execute_failed.value
        self.error_info = error_info
        self.am_err_key = am_err_key

    def __str__(self):
        return self.error_code + self.error_info

    def get_info(self):
        return self.error_info


class SourceScanNoResultException(SourceScanException):
    """
    定义源码扫描无结果异常
    """

    def __init__(self, error_key, error_info=''):
        super().__init__(self)
        self.error_code = SourceScanErrorCode.source_scan_no_result.value
        self.error_key = error_key
        self.error_info = error_info

    def __str__(self):
        return SourceScanErrorInfo.get_en_info(self.error_code) + \
               self.error_info

    def get_error_info(self):
        return self.error_info


class SourceFileNotFoundError(SourceScanException):
    """
    定义源码文件不存在/被删除的异常
    """

    def __init__(self, error_key, error_info=''):
        super().__init__(self)
        self.error_code = SourceScanErrorCode.source_file_not_found.value
        self.error_key = error_key
        self.error_info = error_info

    def __str__(self):
        return SourceScanErrorInfo.get_en_info(self.error_code) + \
               self.error_info

    def get_error_info(self):
        return self.error_info
