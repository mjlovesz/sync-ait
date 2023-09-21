# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import stat
import re
import logging


MAX_SIZE_UNLIMITE = -1  # 不限制，必须显式表示不限制，读取必须传入
MAX_SIZE_LIMITE_CONFIG_FILE = 10 * 1024 * 1024  # 10M 普通配置文件，可以根据实际要求变更
MAX_SIZE_LIMITE_NORMAL_FILE = 4 * 1024 * 1024 * 1024  # 4G 普通模型文件，可以根据实际要求变更
MAX_SIZE_LIMITE_MODEL_FILE = 100 * 1024 * 1024 * 1024  # 100G 超大模型文件，需要确定能处理大文件，可以根据实际要求变更

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

PERMISSION_NORMAL = 0o640  # 普通文件
PERMISSION_KEY = 0o600  # 密钥文件
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def is_legal_path_length(path):
    if len(path) > 4096:
        logger.error(f"file total path length out of range (4096)")
        return False
    dirnames = path.split("/")
    for dirname in dirnames:
        if len(dirname) > 255:
            logger.error(f"file name length out of range (255)")
            return False
    return True


def is_match_path_white_list(path):
    if PATH_WHITE_LIST_REGEX.search(path):
        logger.error(f"path:{path} contains illegal char")
        return False
    return True


def is_legal_args_path_string(path):
    # only check path string
    if not path:
        return True
    if not is_legal_path_length(path):
        return False
    if not is_match_path_white_list(path):
        return False
    return True


class OpenException(Exception):
    pass


class FileStat:
    def __init__(self, file) -> None:
        if not is_legal_path_length(file) or not is_match_path_white_list(file):
            raise OpenException(f"create FileStat failed")
        self.file = file
        self.is_file_exist = os.path.exists(file)
        if self.is_file_exist:
            self.file_stat = os.stat(file)
            self.realpath =  os.path.realpath(file)
        else:
            self.file_stat = None

    @property
    def is_exists(self):
        return self.is_file_exist

    @property
    def is_softlink(self):
        return stat.S_ISLNK(self.file_stat.st_mode) if self.file_stat else False

    @property
    def is_file(self):
        return stat.S_ISREG(self.file_stat.st_mode) if self.file_stat else False

    @property
    def is_dir(self):
        return stat.S_ISDIR(self.file_stat.st_mode) if self.file_stat else False

    @property
    def file_size(self):
        return self.file_stat.st_size if self.file_stat else 0

    @property
    def permission(self):
        return stat.S_IMODE(self.file_stat.st_mode) if self.file_stat else 0o777

    @property
    def owner(self):
        return self.file_stat.st_uid if self.file_stat else -1

    @property
    def group_owner(self):
        return self.file_stat.st_gid if self.file_stat else -1

    @property
    def is_owner(self):
        return self.owner == (os.geteuid() if hasattr(os, "geteuid") else 0)

    @property
    def is_group_owner(self):
        return self.group_owner in (os.getgroups() if hasattr(os, "getgroups") else [0])

    @property
    def is_user_or_group_owner(self):
        return self.is_owner or self.is_group_owner

    @property
    def is_user_and_group_owner(self):
        return self.is_owner and self.is_group_owner

    def is_basically_legal(self, perm='none'):
        if not self.is_exists and perm != 'write':
            logger.error(f"path: {self.file} not exist")
            return False
        if self.is_softlink:
            logger.error(f"path :{self.file} is a symbolic link, considering security, not supported")
            return False
        if not self.is_user_or_group_owner and self.is_exists:
            logger.error(f"current user isn't path:{self.file}'s owner or ownergroup")
            return False
        if perm == 'read':
            if self.permission & READ_FILE_NOT_PERMITTED_STAT > 0:
                logger.error(f"The file {self.file} is group writable, or is others writable.")
                return False
            if not os.access(self.realpath, os.R_OK) or self.permission & stat.S_IRUSR == 0:
                logger.error(f"Current user doesn't have read permission to the file {self.file}.")
                return False
        elif perm == 'write' and self.is_exists:
            if self.permission & WRITE_FILE_NOT_PERMITTED_STAT > 0:
                logger.error(f"The file {self.file} is group writable, or is others writable.")
                return False
            if not os.access(self.realpath, os.W_OK):
                logger.error(f"Current user doesn't have read permission to the file {self.file}.")
                return False
        return True

    def is_legal_file_size(self, max_size):
        if not self.is_file:
            logger.error(f"path: {self.file} is not a file")
            return False
        if self.file_size > max_size:
            logger.error(f"file_size:{self.file_size} byte out of max limit {max_size} byte")
            return False
        else:
            return True

    def is_legal_file_type(self, file_types:list):
        if not self.is_file:
            logger.error(f"path: {self.file} is not a file")
            return False
        for file_type in file_types:
            if os.path.splitext(self.file)[1] == f".{file_type}":
                return True
        logger.error(f"path:{self.file}, file type not in {file_types}")
        return False


def ms_open(file, mode="r", max_size=None, softlink=False, write_permission=PERMISSION_NORMAL, **kwargs):
    file_stat = FileStat(file)

    if file_stat.is_exists and file_stat.is_dir:
        raise OpenException(f"Expecting a file, but it's a folder. {file}")

    if "r" in mode:
        if not file_stat.is_exists:
            raise OpenException(f"No such file or directory {file}")
        if max_size is None:
            raise OpenException(f"Reading files must have a size limit control. {file}")
        if max_size != MAX_SIZE_UNLIMITE and max_size < file_stat.file_size:
            raise OpenException(f"The file size has exceeded the specifications and cannot be read. {file}")

    if "w" in mode:
        if file_stat.is_exists and not file_stat.is_owner:
            raise OpenException(
                f"The file owner is inconsistent with the current process user and is not allowed to write. {file}"
            )
        if file_stat.is_exists:
            os.remove(file)

    if not softlink and file_stat.is_softlink:
        raise OpenException(f"Softlink is not allowed to be opened. {file}")

    if "a" in mode:
        if not file_stat.is_owner:
            raise OpenException(
                f"The file owner is inconsistent with the current process user and is not allowed to write. {file}"
            )
        if file_stat.permission != (file_stat.permission & write_permission):
            os.chmod(file, file_stat.permission & write_permission)

    flags = os.O_RDONLY
    if "+" in mode:
        flags = flags | os.O_RDWR
    elif "w" in mode or "a" in mode or "x" in mode:
        flags = flags | os.O_WRONLY

    if "w" in mode or "x" in mode:
        flags = flags | os.O_TRUNC | os.O_CREAT
    if "a" in mode:
        flags = flags | os.O_APPEND | os.O_CREAT
    return os.fdopen(os.open(file, flags, mode=write_permission), mode, **kwargs)
