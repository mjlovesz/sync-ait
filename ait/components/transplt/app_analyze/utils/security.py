import os
import platform

MAX_JSON_FILE_SIZE = 10 * 1024 ** 2
WINDOWS_PATH_LENGTH_LIMIT = 200
LINUX_FILE_NAME_LENGTH_LIMIT = 200


class SoftLinkCheckException(Exception):
    pass


def islink(path):
    path = os.path.abspath(path)
    return os.path.islink(path)


def check_path_length_valid(path):
    path = os.path.realpath(path)
    if platform.system().lower() == 'windows':
        return len(path) <= WINDOWS_PATH_LENGTH_LIMIT
    else:
        return len(os.path.basename(path)) <= LINUX_FILE_NAME_LENGTH_LIMIT


def check_input_file_valid(input_path, max_file_size=MAX_JSON_FILE_SIZE):
    if islink(input_path):
        raise SoftLinkCheckException("Input path doesn't support soft link.")

    input_path = os.path.realpath(input_path)
    if not os.path.exists(input_path):
        raise ValueError('Input file %s does not exist!' % input_path)

    if not os.access(input_path, os.R_OK):
        raise PermissionError('Input file %s is not readable!' % input_path)

    if not check_path_length_valid(input_path):
        raise ValueError('The real path or file name of input is too long.')

    if os.path.getsize(input_path) > max_file_size:
        raise ValueError(f'The file is too large, exceeds {max_file_size // 1024 ** 2}MB')
