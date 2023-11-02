import os

from .logger import logger


def check_permission(file):
    if not os.path.exists(file):
        logger.error(f"path: {file} not exist, please check if file or dir is exist")
        return False
    if os.path.islink(file):
        logger.error(f"path :{file} is a soft link, not supported, please import file(or directory) directly")
        return False
    return True
