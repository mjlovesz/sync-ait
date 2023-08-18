import os
import stat


MAX_SIZE_UNLIMITE = -1  # 就是不限制。必须显示表示不限制，读取必须传入
MAX_SIZE_LIMITE_CONFIG_FILE = 10 * 1024 * 1024  # 10M 普通配置文件，可以根据实际要求变更
MAX_SIZE_LIMITE_MODEL_FILE = 4 * 1024 * 1024 * 1024  # 4G 普通模型文件，可以根据实际要求变更
MAX_SIZE_LIMITE_MODEL_FILE = 100 * 1024 * 1024 * 1024  # 100G 超大模型文件，需要确定您确实能处理这么大的文件，可以根据实际要求变更

PERMISSION_NORMAL = 0o640  # 普通文件
PERMISSION_KEY = 0o600  # 密钥文件


class OpenException(Exception):
    pass


class FileStat:
    def __init__(self, file) -> None:
        self.is_file_exist = os.path.exists(file)
        if self.is_file_exist:
            self.file_stat = os.stat(file)
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
    def is_owner(self):
        return self.owner == (os.geteuid() if hasattr(os, "geteuid") else 0)


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
    elif "w" in mode or "a" in mode:
        flags = flags | os.O_WRONLY

    if "w" in mode:
        flags = flags | os.O_TRUNC | os.O_CREAT
    if "a" in mode:
        flags = flags | os.O_APPEND | os.O_CREAT
    return os.fdopen(os.open(file, flags, mode=write_permission), mode, **kwargs)
