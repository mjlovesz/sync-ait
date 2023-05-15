# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import os
import re
from enum import Enum, unique
from collections import namedtuple
import platform


@unique
class ScannerType(Enum):
    """
    扫描器枚举类型
    """
    INVALID_SCANNER = -1  # 无效的扫描器种类
    CPP_SCANNER = 0  # C/C++源代码扫描器
    MAKEFILE_SCANNER = 1  # Makefile文件扫
    CMAKE_SCANNER = 2  # cmakelist文件扫描


@unique
class ReporterType(Enum):
    """
    迁移报告格式枚举类型
    """
    INVALID_REPORTER = -1  # 无效的报告种类
    CSV_REPORTER = 0  # csv(xlsx)报告
    JSON_REPORTER = 1  # json格式供Django读取


@unique
class InputType(Enum):
    """
    input type
    """
    CMD_LINE = 'cmd'
    RESTFUL = 'restful'


class KitConfig:
    MACRO_PATTERN = re.compile(r'(OpenCV|CUDA|NVJPEG|DALI|CVCUDA)')
    LIBRARY_PATTERN = re.compile(
        r'nvjpeg_static|nvjpeg2k_static|avdevice|avfilter|avformat|avcodec|swresample|swscale|avutil|postproc|'
        r'libnvjpeg_static|libnvjpeg2k_static|libavdevice|libavfilter|libavformat|libavcodec|libswresample|libswscale|'
        r'libavutil|libpostproc|libnvcuvid|libnvidia-encode|libcvcuda|libnvcv_types|libnvcv_types')
    FILE_PATTERN = re.compile(r'opencv.hpp|opencv2')
    UNKNOWN_PATTERN = re.compile(r'opencv|cuda|dali|nvjpeg|ffmpeg')

    THREAD_NUM = 3

    ARCH = platform.machine()
    LIB_CLANG_PATH = f'/usr/lib/{ARCH}-linux-gnu/libclang-14.so'
    HEADERS_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'headers'))
    INCLUDES = {
        'cuda': '',
        'opencv': f'{HEADERS_FOLDER}/opencv/include/opencv4',
        'tensorrt': '',
    }

    # 'make', 'automake'
    VALID_CONSTRUCT_TOOLS = ['cmake']
    PORTING_CONTENT = """ait transplt
            [-h] [-s source] 
            [-t tools] 
            [-l {DEBUG,INFO,WARN,ERR}] 
            [-f report_type]\n"""

    SOURCE_DIRECTORY = ''
    PROJECT_TIME = ''

    VALID_REPORT_TYPE = ['csv', 'json']
    API_MAP = '../config/mxBase_API_MAP.xlsx'
    EXCEPT_API = ['', 'NAMESPACE_REF']

    CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    # lib_name: [namespace, cuda_include, cuda_namespace]，后两者用于分析基于CUDA加速的接口
    # cuda_include参考示例：
    # OpenCV-CUDA
    # "opencv2/core/cuda.hpp", "opencv2/cudaarithm.hpp", "opencv2/cudaimgproc.hpp", "opencv2/cudabgsegm.hpp",
    # "opencv2/cudawarping.hpp", "opencv2/cudaobjdetect.hpp", "opencv2/cudafilters.hpp", "opencv2/cudastereo.hpp",
    # "opencv2/cudafeatures2d.hpp", "opencv2/xfeatures2d/cuda.hpp", "opencv2/cudacodec.hpp",
    # "opencv2/core/cuda_types.hpp", "opencv2/core/cuda_stream_accessor.hpp", "opencv2/core/cuda.inl.hpp"
    # FFmpeg-CUDA
    # "libavcodec/nvenc.h"
    acc_libs = {
        # OpenCV
        '/opencv2/': ['cv', '/cuda', ['cuda', 'gpu']],
        # FFMPEG: https://github.com/FFmpeg/FFmpeg
        '/libavcodec/': ['', '/nv', None],
        '/libavfilter/': ['', ['/cuda/', '_cuda'], None],
        'libavformat': '',
        '/libavdevice/': '',
        '/libavutil/': ['', ['/cuda_', '_cuda/', '_cuda_'], None],
        '/libswresample/': '',
        '/libpostproc/': '',
        '/libswscale/': '',
        # CUDA samples: https://github.com/NVIDIA/CUDALibrarySamples
        CUDA_HOME: ['', 1, ''],
        # nvJPEG samples: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/nvJPEG
        'nvjpeg': ['', 1, ''],
        # DALI: https://github.com/NVIDIA/DALI
        'dali': ['dali', 1, ''],
        # CV-CUDA
        '/cvcuda': ['cvcuda', 1, '']
    }
    LEVEL = 'small'  # parse level: 'large'
    PRINT_DETAIL = False
    TOLERANCE = 4  # code diag level: {'ignored':0, 'info':1, 'warning':2, 'error':3, 'fatal':4}


@unique
class FileType(Enum):
    INVALID_FILE_TYPE = -1
    C_SOURCE_FILE = 0
    MAKEFILE = 1
    PURE_ASSEMBLE = 2
    CMAKE_LISTS = 3
    AUTOMAKE_FILE = 4
    PYTHON_FILE = 5


# 定义源码迁移的返回结果的数据结构
PortingResult = namedtuple(
    'PortingResult', ['file_path',
                      'file_type',
                      'code_range',
                      'total_rows',
                      'category',
                      'keyword',
                      'suggestion',
                      'description',
                      'suggestion_type',
                      'replacement']
)
