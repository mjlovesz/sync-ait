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
    cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH')
    if cmake_prefix_path:
        cmake_prefix_path = filter(bool, cmake_prefix_path.split(':'))  # 去空字符穿
        for p in cmake_prefix_path:
            if not p.startswith('/usr/'):
                sys_paths.append(p)


def _opencv_dir_path(sys_paths):
    # //The directory containing a CMake configuration file for OpenCV.
    opencv_dir = os.environ.get('OpenCV_DIR')
    if opencv_dir and not opencv_dir.startswith('/usr/'):
        opencv_dir = os.path.normpath(opencv_dir + '/../../../')
        sys_paths.append(opencv_dir)


def _cuda_path(sys_paths):
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if not cuda_home.startswith('/usr/'):
        sys_paths.append(cuda_home)


def _opencv_include_path(sys_paths):
    if not KitConfig.opencv_include_path.startswith('/usr/'):
        sys_paths.append(KitConfig.opencv_include_path)
