from utils.log_util import logger
from porting.app import init_args, start_scan_kit


def modify_cfg_character(level):
    """修改变量"""
    if level == "ERR":
        level = "ERROR"

    elif level == "WARN":
        level = "WARNING"

    return level


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    args = init_args()
    logger.setLevel(modify_cfg_character(args.log_level))

    start_scan_kit(args)
