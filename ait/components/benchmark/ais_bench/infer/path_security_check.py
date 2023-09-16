import os
import sys
import re
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def path_length_check(path):
    if len(path) > 4096:
        logger.error(f"file total path length out of range (4096)")
        return False
    dirnames = path.split("/")
    for dirname in dirnames:
        if len(dirname) > 255:
            logger.error(f"file name length out of range (255)")
            return False
    return True


def path_white_list_check(path):
    regex = re.compile(r"[^_A-Za-z0-9/.-]")
    if regex.search(path):
        logger.error(f"path:{path} contains illegal char")
        return False
    return True


def path_exist_check(path):
    if not os.path.exists(os.path.realpath(path)):
        logger.error(f"path:{path} not exist")
        return False
    return True


def path_symbolic_link_check(path):
    if os.path.islink(path):
        logger.error(f"path:{path} is a symbolic link, considering security, not supported")
        return False
    return True


def path_owner_correct_check(path):
    path_stat = os.stat(path)
    current_owner_id = os.getuid()
    current_group_id = os.getgid()
    path_owner_id = path_stat.st_uid
    path_group_id = path_stat.st_gid
    if current_owner_id == path_owner_id and current_group_id == path_group_id:
        return True
    else:
        logger.error(f"current user isn't path:{path}'s owner and ownergroup")
        return False