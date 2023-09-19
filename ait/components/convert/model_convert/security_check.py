import os
import stat
import sys
import re

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH

input_file_args = ['model', 'weight', 'singleop', 'insert_op_conf', 'op_name_map', 'fusion_switch_file',
                   'compression_optimize_conf', 'op_debug_config']
input_dir_args = ['mdl_bank_path', 'op_bank_path', 'debug_dir', 'op_compiler_cache_dir', 'model_path']
output_file_args = ['output', 'json', ]


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def is_endswith_extensions(path, extensions):
    result = False
    if isinstance(extensions, (list, tuple)):
        for extension in extensions:
            if path.endswith(extension):
                result = True
                break
    elif isinstance(extensions, str):
        result = path.endswith(extensions)
    return result


def get_valid_path(path, extensions=None):
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")

    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be soft link: {}.".format(path))

    real_path = os.path.realpath(path)

    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise ValueError("The length of filename should be less than 256.")
    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    if extensions and not is_endswith_extensions(path, extensions):  # Check whether the file name endswith extension
        raise ValueError("The filename {} doesn't endswith \"{}\".".format(path, extensions))

    return real_path


def check_write_directory(dir_name, check_user_stat=True):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise ValueError("The file writen directory {} doesn't exists.".format(dir_name))

    file_stat = os.stat(real_dir_name)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file writen directory {} doesn't belong to the current user or group.".format(dir_name))
    if not os.access(real_dir_name, os.W_OK):
        raise ValueError("Current user doesn't have writen permission to file writen directory {}.".format(dir_name))


def get_valid_read_path(path, extensions=None, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path, extensions)
    if not is_dir and not os.path.isfile(real_path):
        raise ValueError("The path {} doesn't exists or not a file.".format(path))
    if is_dir and not os.path.isdir(real_path):
        raise ValueError("The path {} doesn't exists or not a directory.".format(path))

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file {} doesn't belong to the current user or group.".format(path))
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise ValueError("The file {} is group writable, or is others writable.".format(path))
    if not os.access(real_path, os.R_OK) or file_stat.st_mode & stat.S_IRUSR == 0:  # At least been 400
        raise ValueError("Current user doesn't have read permission to the file {}.".format(path))
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError("The file {} exceeds size limitation of {}.".format(path, size_max))
    return real_path


def get_valid_write_path(path, extensions=None, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path, extensions)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)

    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise ValueError("The file {} exist and is a directory.".format(path))
        if check_user_stat and os.stat(real_path).st_uid != os.getuid():  # Has to be exactly belonging to current user
            raise ValueError("The file {} doesn't belong to the current user.".format(path))
        if check_user_stat and os.stat(real_path).st_mode & WRITE_FILE_NOT_PERMITTED_STAT > 0:
            raise ValueError("The file {} permission for others is not 0, or is group writable.".format(path))
        if not os.access(real_path, os.W_OK):
            raise ValueError("The file {} exist and not writable.".format(path))
    return real_path
