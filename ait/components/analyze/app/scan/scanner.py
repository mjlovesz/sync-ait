import logging

logging.raiseExceptions = False


class Scanner:
    """
    Scanner类作为扫描器的基类存在，仅定义必要的属性和方法接口。
    """
    __slots__ = ['files', 'porting_results', 'name', 'pool_numbers']

    def __init__(self, files):
        self.files = files
        self.porting_results = {}

    def do_scan(self):
        raise NotImplementedError('{} must implement do_scan method!'.format(self.__class__))
