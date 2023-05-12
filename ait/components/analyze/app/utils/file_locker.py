# Copyright 2023 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference:
# - https://docs.python.org/zh-cn/3/library/fcntl.html
# - https://docs.python.org/zh-cn/3/library/msvcrt.html
# - https://docs.python.org/zh-cn/3/library/ctypes.html#ctypes.WinDLL
# - https://juejin.cn/post/6870689230440529927
# - https://zhuanlan.zhihu.com/p/354383209
"""
用法1：
with open('x.py', 'w') as fd:
    lock(fd)
用法2：
fd = open('x.py', 'w')
lock(fd)
# ...
unlock(fd)
"""
import platform

if platform.system() != 'Windows':
    import fcntl

    IS_UNIX = True
    LOCK_FILE = '/tmp/file.lock'
else:
    import msvcrt

    IS_UNIX = False
    LOCK_FILE = 'C:\\file.lock'

NBYTES = 1
LOCK_EX = 2
LOCK_NB = 4


def _lock_nb_mode(file_desc):
    if IS_UNIX:
        try:
            fcntl.flock(file_desc, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    else:
        try:
            msvcrt.locking(file_desc.fileno(), msvcrt.LK_NBLCK, NBYTES)
        except OSError:
            return False
        file_desc.seek(0)

    return True


def _lock_ex_mode(file_desc):
    if IS_UNIX:
        fcntl.flock(file_desc, fcntl.LOCK_EX)
    else:
        msvcrt.locking(file_desc.fileno(), msvcrt.LK_LOCK, NBYTES)
        file_desc.seek(0)
    return True


def lock(file_desc, mode=LOCK_EX):
    """同一进程内对同一文件重复加锁，不同进程对同一个文件重复加锁，会阻塞或返回False。"""
    if mode == LOCK_NB:
        return _lock_nb_mode(file_desc)
    else:
        return _lock_ex_mode(file_desc)


def unlock(file_desc):
    if IS_UNIX:
        fcntl.flock(file_desc, fcntl.LOCK_UN)
    else:
        msvcrt.locking(file_desc.fileno(), msvcrt.LK_UNLCK, 1)
    return True
