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
    CSV_REPORTER = 0  # csv报告
    JSON_REPORTER = 1  # json格式供Django读取
    HTML_REPORTER = 2  # html报告


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

    thread_num = 3

    arch = platform.machine()
    lib_clang_path = f'/usr/lib/{arch}-linux-gnu/libclang-6.0.so'
    includes = {
        'cuda': '',
        'opencv': '',
        'tensorrt': '',
    }

    # TODO: 'make', 'automake'
    valid_construct_tools = ['cmake']
    porting_content = """porting-advisor
            [-h] [-s source] 
            [-t tools] 
            [-l {DEBUG,INFO,WARN,ERR}] 
            [-f report_type]\n"""

    source_directory = ''
    project_time = ''

    valid_report_type = ['csv', 'json']
    api_map = '../config/mxBase_API_MAP.xlsx'
    except_api = ['', 'NAMESPACE_REF']

    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
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
        cuda_home: ['', 1, ''],
        # nvJPEG samples: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/nvJPEG
        'nvjpeg': ['', 1, ''],
        # DALI: https://github.com/NVIDIA/DALI
        'dali': ['dali', 1, ''],
        # CV-CUDA
        '/cvcuda': ['cvcuda', 1, '']
    }
    level = 'small'  # 'large'
    print_detail = False


@unique
class FileType(Enum):
    INVALID_FILE_TYPE = -1
    C_SOURCE_FILE = 0
    MAKEFILE = 1
    PURE_ASSEMBLE = 2
    CMAKE_LISTS = 3
    AUTOMAKE_FILE = 4
    FORTRAN_FILE = 5
    PYTHON_FILE = 6
    GOLANG_FILE = 7
    JAVA_FILE = 8
    SCALA_FILE = 9


@unique
class PortingCategory(Enum):
    """
    扫描出的迁移项的类型.
    NOTE: AarchSpecific这个值必须放在scanner_factory
    self.pattern构造的最后一个位置，
    并根据那个位置索引修正这个枚举的位置, 非常重要！！！
    """
    INVALID_CATEGORY = -1  # 无效的迁移项类型
    INTRINSICS = 0  # intrinsics内联函数
    COMPILER_MACRO = 1  # 编译器宏
    ATTRIBUTE = 2  # 编译器attribute
    COMPILER_BUILTIN = 3  # 编译器内建函数
    COMPILER_OPTION = 4  # 编译器选项
    BUILTIN_ASSEMBLES = 5  # 嵌入式汇编
    LIBS = 6  # 扫描出动态链接库
    COMPILER_OPTION_SPECIAL = 7  # 特殊编译器选项
    MODULE_FUNCTION = 8  # modulefuntion
    AARCH_SPECIFIC = 9  # aarch平台独有，根据keep-going的值来判断是否提前结束扫描
    PURE_ASSEMBLES = 10  # 纯汇编文件
    AUTOMAKE_FILE = 11  # automake文件
    FORTRAN_COMPILER_OPTION = 12  # gfortran的编译选项
    FORTRAN_BUILTIN = 13  # Fortran的内建函数
    FORTRAN_GRAMMAR = 14  # Fortran的语法
    PRECOMPILED_MACRO = 15  # 预编译宏
    PYTHON_LIBRARY = 16  # python文件扫描出的so
    PYTHON_LOAD_LIBRARY = 17  # python文件扫描出加载so所在行
    JAVA_LIBRARY = 18  # java文件扫描出的so
    JAVA_LOAD_LIBRARY = 19  # java文件扫描出加载so所在行
    SCALA_LIBRARY = 20  # scala文件扫描出的so
    SCALA_LOAD_LIBRARY = 21  # scala文件扫描出加载so所在行
    FORTRAN_MODULE_FILE = 22  # Fortran module文件编译出的.mod文件
    MIX_FUNCTION_NOT_MATCH = 23  # C和Fortran函数互调参数和返回值不匹配


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
