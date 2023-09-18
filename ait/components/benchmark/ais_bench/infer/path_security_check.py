import os
import sys
import re
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def args_path_input_check(path, perm_list:list):
    # check path as input path
    if not path:
        return True
    if not path_length_check(path):
        return False
    if not path_white_list_check(path):
        return False
    if not path_exist_check(path):
        return False
    for perm in perm_list:
        if not os.access(path, perm):
            logger.error(f"file {path} don't have right permission")
            return False
    if not path_symbolic_link_check(path):
        return False
    if not path_owner_correct_check(path):
        return False
    return True


def args_path_output_check(path):
    # check path as output path
    if not path:
        return True
    if not path_length_check(path):
        return False
    if not path_white_list_check(path):
        return False
    return True


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


def path_file_size_check(path, max_size):
    file_size = os.path.getsize(path)
    if file_size > max_size:
        logger.error(f"acl_json_file_size:{file_size} byte out of max limit {max_size} byte")
        return False
    else:
        return True


def path_file_type_check(path, file_type:str):
    if os.path.splitext(path)[1] != f".{file_type}":
        logger.error(f"acl_json_path:{path} is not a .{file_type} file")
        return False
    else:
        return True