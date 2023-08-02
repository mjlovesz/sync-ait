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

from app_analyze.utils.clang_finder import get_lib_clang_path


_sep = os.path.sep


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
    # 1. 工具运行相关配置
    ARCH = platform.machine()

    @classmethod
    def lib_clang_path(cls):
        return get_lib_clang_path()

    CXX_STD = 'c++17'  # c++11、c++14、c++17、c++20等，或者None，表示使用clang默认值

    # 'make', 'automake'
    VALID_REPORT_TYPE = ['csv', 'json']
    VALID_CONSTRUCT_TOOLS = ['cmake']
    PORTING_CONTENT = """ait transplt
                [-h] [-s source] 
                [-t tools] 
                [-l {DEBUG,INFO,WARN,ERR}] 
                [-f report_type]\n"""

    SOURCE_DIRECTORY = ''
    PROJECT_TIME = ''

    # 2. 加速库相关配置
    # a.加速库名
    OPENCV = 'OpenCV'
    FFMPEG = 'FFmpeg'
    CUDA = 'CUDA'
    DALI = 'DALI'
    CVCUDA = 'CVCUDA'
    TENSORRT = 'TensorRT'
    CODEC = 'Codec'

    # b.加速库路径
    HEADERS_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'headers'))
    INCLUDES = {
        OPENCV: os.path.join(HEADERS_FOLDER, 'opencv', 'include', 'opencv4'),
        FFMPEG: os.path.join(HEADERS_FOLDER, 'ffmpeg', 'include'),
        CUDA: os.path.join(HEADERS_FOLDER, 'cuda', 'include'),
        CVCUDA: os.path.join(HEADERS_FOLDER, 'cvcuda', 'include'),
        TENSORRT: os.path.join(HEADERS_FOLDER, 'tensorrt', 'include'),
        CODEC: os.path.join(HEADERS_FOLDER, 'codec', 'include'),
    }
    # c.如果用户已经安装，选取用户安装的路径，否则选取默认配置
    OPENCV_HOME = os.environ.get('OPENCV_HOME', INCLUDES.get(OPENCV, None))
    CUDA_HOME = os.environ.get('CUDA_HOME', INCLUDES.get(CUDA, None))
    CVCUDA_HOME = os.environ.get('CVCUDA_HOME', INCLUDES.get(CVCUDA, None))
    TENSORRT_HOME = os.environ.get('TENSORRT_HOME', INCLUDES.get(TENSORRT, None))
    CODEC_HOME = os.environ.get('CODEC_HOME', INCLUDES.get(CODEC, None))

    # d.C++加速库模式匹配
    # 格式如下，第0/1/2可为list，第1/2用于分析基于CUDA加速的接口。
    # namespace, cuda_include, cuda_namespace, lib_name
    #
    # cuda使能头文件示例：
    # OpenCV-CUDA：
    # "opencv2/core/cuda.hpp", "opencv2/cudaarithm.hpp", "opencv2/cudaimgproc.hpp", "opencv2/cudabgsegm.hpp",
    # "opencv2/cudawarping.hpp", "opencv2/cudaobjdetect.hpp", "opencv2/cudafilters.hpp", "opencv2/cudastereo.hpp",
    # "opencv2/cudafeatures2d.hpp", "opencv2/xfeatures2d/cuda.hpp", "opencv2/cudacodec.hpp",
    # "opencv2/core/cuda_types.hpp", "opencv2/core/cuda_stream_accessor.hpp", "opencv2/core/cuda.inl.hpp"
    # FFmpeg-CUDA：
    # "libavcodec/nvenc.h"
    ACC_LIBS = {
        # OpenCV
        OPENCV_HOME: ['cv', _sep + 'cuda', ['cuda', 'gpu'], OPENCV],
        # FFmpeg
        _sep + 'libavcodec' + _sep: ['', _sep + 'nv', '', FFMPEG],
        _sep + 'libavfilter' + _sep: ['', [_sep + 'cuda' + _sep, '_cuda'], '', FFMPEG],
        'libavformat': ['', '', '', FFMPEG],
        _sep + 'libavdevice' + _sep: ['', '', '', FFMPEG],
        _sep + 'libavutil' + _sep: ['', [_sep + 'cuda_', '_cuda' + _sep, '_cuda_'], '', FFMPEG],
        _sep + 'libswresample' + _sep: ['', '', '', FFMPEG],
        _sep + 'libpostproc' + _sep: ['', '', '', FFMPEG],
        _sep + 'libswscale' + _sep: ['', '', '', FFMPEG],
        # DALI
        'dali': ['dali', 1, '', DALI],
        # CUDA
        # nvJPEG
        CUDA_HOME: ['', 1, '', CUDA],  # 含nvJPEG等
        # CV-CUDA
        CVCUDA_HOME: [['nvcv', 'cvcuda'], 1, '', CVCUDA],
        TENSORRT_HOME: [['nvinfer1'], 1, '', TENSORRT],
        CODEC_HOME: ['', 1, '', CODEC],
    }

    # e.API映射表，文件名第一个'_'前为加速库名；内部工作表/Sheet名以'-APIMap'结尾，其他工作表会被忽略。
    API_MAP_FOLDER = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, 'config'))
    API_MAP = {
        OPENCV: os.path.join(API_MAP_FOLDER, 'mxBase_OpenCV_API_MAP.xlsx'),
        FFMPEG: os.path.join(API_MAP_FOLDER, 'DVPP_FFMPEG_API_MAP.xlsx'),
        CUDA: os.path.join(API_MAP_FOLDER, 'ACL_CUDA_API_MAP.xlsx'),
        CVCUDA: os.path.join(API_MAP_FOLDER, 'mxBase_CVCUDA_API_MAP.xlsx'),
        TENSORRT: os.path.join(API_MAP_FOLDER, 'ACLMDL_TRT_API_MAP.xlsx'),
        CODEC: os.path.join(API_MAP_FOLDER, 'DVPP_Codec_API_MAP.xlsx'),
    }

    # 3.CMake加速库模式匹配
    MACRO_PATTERN = re.compile(r'(OpenCV|CUDA|NVJPEG|DALI|CVCUDA)')
    LIBRARY_PATTERN = re.compile(
        r'nvjpeg_static|nvjpeg2k_static|avdevice|avfilter|avformat|avcodec|swresample|swscale|avutil|postproc|'
        r'cvcuda|nvcv_types|'
        r'libnvjpeg_static|libnvjpeg2k_static|libavdevice|libavfilter|libavformat|libavcodec|libswresample|libswscale|'
        r'libavutil|libpostproc|libnvcuvid|libnvidia-encode|libcvcuda|libnvcv_types')
    FILE_PATTERN = re.compile(r'opencv.hpp|opencv2')
    KEYWORD_PATTERN = re.compile(r'opencv|cuda|dali|nvjpeg|ffmpeg|cudart')

    # 4. Report字段，含扫描分析和API Map字段
    # a.源于扫描分析
    ACC_API = 'AccAPI'  # 三方加速库API
    CUDA_EN = 'CUDAEnable'  # 是否CUDA
    LOCATION = 'Location'  # 调用三方加速库API的位置
    CONTEXT = 'Context(形参 | 实参 | 来源代码 | 来源位置)'  # 三方加速库API参数及上下文
    ACC_LIB = 'AccLib'  # API所属三方加速库
    # b.源于API MAP
    ASCEND_LIB = 'AscendLib'  # 推荐的昇腾API所属库
    ASCEND_API = 'AscendAPI'  # 昇腾API
    DESC = 'Description'  # API描述
    WORKLOAD = 'Workload(人/天)'  # 迁移工作量（人/天）
    PARAMS = 'Params(Ascend:Acc)'  # 昇腾API和三方加速库API形参对应关系
    ASCEND_LINK = 'AscendAPILink'  # 昇腾API文档链接
    ACC_LINK = 'AccAPILink'  # 三方加速库API文档链接
    # c.可选报告字段
    OPT_REPORT_KEY = {
        DESC: True,
        CONTEXT: True,
        ACC_LIB: True,
        ASCEND_LIB: True,
        PARAMS: False,
        ACC_LINK: True,
        ASCEND_LINK: True,
    }
    EXCEPT_API = ['']  # 扫描时忽略的API
    DEFAULT_WORKLOAD = 0.1  # 无映射关系/未设置工作量的API的默认工作量

    # 5.可选配置
    LEVEL = 'small'  # parse level: 'large'
    TOLERANCE = 4  # code diag level: {'ignored':0, 'info':1, 'warning':2, 'error':3, 'fatal':4}
    CURSOR_DEPTH = 100


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
