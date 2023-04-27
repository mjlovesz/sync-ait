import os
from common.kit_config import KitConfig


def get_sys_path():
    sys_paths = ['/usr/']

    _cmake_prefix_path(sys_paths)
    _opencv_dir_path(sys_paths)
    _cuda_path(sys_paths)
    _opencv_include_path(sys_paths)

    return sys_paths


def _cmake_prefix_path(sys_paths):
    CMAKE_PREFIX_PATH = os.environ.get('CMAKE_PREFIX_PATH')
    if CMAKE_PREFIX_PATH:
        cmake_prefix_path = filter(bool, CMAKE_PREFIX_PATH.split(':'))  # 去空字符穿
        for p in cmake_prefix_path:
            if not p.startswith('/usr/'):
                sys_paths.append(p)


def _opencv_dir_path(sys_paths):
    # //The directory containing a CMake configuration file for OpenCV.
    OpenCV_DIR = os.environ.get('OpenCV_DIR')
    if OpenCV_DIR and not OpenCV_DIR.startswith('/usr/'):
        opencv_dir = os.path.normpath(OpenCV_DIR + '/../../../')
        sys_paths.append(opencv_dir)


def _cuda_path(sys_paths):
    CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if not CUDA_HOME.startswith('/usr/'):
        sys_paths.append(CUDA_HOME)


def _opencv_include_path(sys_paths):
    if not KitConfig.opencv_include_path.startswith('/usr/'):
        sys_paths.append(KitConfig.opencv_include_path)
