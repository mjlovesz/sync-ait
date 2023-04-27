import argparse
import os
from common.kit_config import KitConfig
from scan.scan_api import ScanApi


def init_args():
    """
    设置工具的命令行参数
    :return: 接收的命令行参数对象
    """
    parser = argparse.ArgumentParser()

    # 添加需要进行扫描的目录列表，逗号分隔的情况下只有一个列表元素
    parser.add_argument('-s', '--source', dest='source', help='directories of source folder')

    # 添加输出格式选项，当前默认和仅支持xlsx
    parser.add_argument('-f', '--report-type', dest='report_type', default='csv',
                        help='specify output report type. Only xlsx is supported.')

    # 添加日志级别开关，默认级别是INFO，只有添加-d后才能输出ERR级别日志
    parser.add_argument('-l', '--log-level', dest='log_level',
                        choices=['DEBUG', 'INFO', 'WARN', 'ERR'],
                        default='INFO',
                        help='specify log level. default is '
                             'INFO. choices from: DEBUG, INFO, WARN, ERR')

    parser.add_argument('-t', '--tools', dest='tools',
                        choices=KitConfig.valid_construct_tools,
                        default='cmake',
                        help='specify construction. default is cmake.')

    return parser.parse_args()


def start_scan_kit(args):
    if not os.path.exists(args.source):
        raise Exception("Source directory is not existed!")
    args.source = os.path.abspath(args.source)

    KitConfig.source_directory = args.source
    scan_api = ScanApi()
    scan_api.scan_source(args)
