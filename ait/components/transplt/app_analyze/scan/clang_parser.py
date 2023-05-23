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

"""
Possible methods:
1. LLVM python bindings: https://github.com/llvm/llvm-project/tree/main/clang/bindings/python
2. http://gccxml.github.io/HTML/Index.html

libclang(mirrored LLVM python bindings): https://readthedocs.org/projects/libclang/
clang(mirrored LLVM python bindings): https://pypi.org/project/clang/
libclang case: https://eli.thegreenplace.net/2011/07/03/parsing-c-in-python-with-clang/

Note that the package libclang on PyPi bundles the prebuilt shared library, but may be incompatible with the system.
The package clang on PyPi doesn't bundle the prebuilt shared library, and the clang tool should be installed.
"""
import logging
import re
import os
import time

from clang.cindex import Index, CursorKind, TranslationUnit, Config

from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.io_util import IOUtil
from app_analyze.utils.log_util import logger
from app_analyze.utils.lib_util import get_sys_path
from app_analyze.scan.clang_utils import helper_dict, filter_dict, Info,\
    get_attr, get_children, skip_implicit, auto_match
from app_analyze.scan.clang_utils import read_cursor

SYS_PATH = get_sys_path()
SCANNED_FILES = list()
RESULTS = list()
MACRO_MAP = dict()
# set the config
if not Config.loaded:
    # 或指定目录：Config.set_library_path("/usr/lib/x86_64-linux-gnu")
    Config.set_library_file(KitConfig.LIB_CLANG_PATH)


def get_diag_info(diag):
    return {'info': diag.format(),
            'fixits': list(diag.fixits)}


def get_ref_def(cursor):
    """
    C++标准/内置的Node无referenced和definition，如UNEXPOSED_EXPR/XXX_LITERAL/XXX_OPERATOR/XXX_STMT，部分CALL_EXPR。

    声明的类/函数/变量等无referenced。
    方法/函数调用无definition，如CALL_EXPR。
    部分构造函数调用无definition。
    """
    ref = cursor.referenced
    decl = cursor.get_definition()
    result = list()
    if ref is not None:
        result.extend([f'ref: {ref.displayname}', f'{ref.location}'])  # {ref.extent.start.line}
    if decl is not None:
        result.extend([f'def: {decl.displayname}', f'{decl.location}'])
    return result


def cuda_enabled(file, include, namespace=None):
    """判断该文件是否为加速库内cuda相关文件。"""
    if not isinstance(include, list):
        include = [include]
    for x in include:
        if x == 1 or x in file:
            return True
    return False


def add_namespace(cursor, namespaces):
    """获取未显示的命名空间"""
    if not namespaces or not cursor.referenced:
        return ''
    if not isinstance(namespaces, list):
        namespaces = [namespaces]
    usr = cursor.referenced.get_usr()
    index = usr.find(cursor.referenced.spelling)
    if index == -1:
        return ''
    nsc = re.findall(r'(?:@N@\w+){1,1000}', usr[:index])
    nss = ['::'.join(x[3:].split('@N@')) for x in nsc]
    for namespace in namespaces:
        for ns in reversed(nss):
            if namespace in ns:  # namespace可能是pattern，不是完整namespace
                return ns
    return ''


def in_acc_lib(file, cursor):
    """判断该文件是否为加速库文件。"""
    if not file:
        return False, False, ''
    for lib, v in KitConfig.ACC_LIBS.items():
        if lib in file:
            if not v:
                cuda_en = False
                add_ns = ''
            else:
                cuda_en = cuda_enabled(file, v[1])
                add_ns = add_namespace(cursor, v[0])
            return True, cuda_en, add_ns
    return False, False, ''


def filter_acc(cursor):
    hit = False
    result_type = None
    spelling = None
    api = None
    definition = None
    source = None
    cuda_en = False
    ns = ''

    if cursor.kind.name in helper_dict:  # 用于提前对节点进行处理，比如VAR_DECL的命名空间、FUNCTIONPROTO的参数等
        result_type, spelling, api, definition, source = helper_dict[cursor.kind.name](cursor)
    if cursor.kind.name in filter_dict:
        result_type, spelling, api, definition, source = filter_dict[cursor.kind.name](cursor)
        hit, cuda_en, ns = in_acc_lib(source, cursor)

    if ns and api and f'{ns}::' not in api:
        core = api.split('::')[-1]
        api = f'{ns}::{core}'
    return hit, Info(result_type, spelling, api, definition, source), cuda_en


def get_includes(tu):
    """
    获取FileInclusion，没检测到/不存在对应库文件时，不生效。如：
        #include <fstream>
        #include <opencv2>
        #include <person.hpp>
    """
    includes = tu.get_includes()
    logger.debug('depth, include.name, location.line, location.column, source')
    srcs = list()
    for x in includes:
        if x.depth < 2:  # 1,2,3...
            logger.debug(x.depth, x.include.name, get_attr(x, 'location.line'),
                         get_attr(x, 'location.column'), x.source)
            srcs.append(x.include.name)
    return srcs


def is_usr_code(file):
    usr_code = True
    if not file or any(file.startswith(p) for p in SYS_PATH):
        usr_code = False
    return usr_code


def macro_map(cursor):
    """过滤并保存宏定义到字典中，主要用于标识符重命名场景。

    如：#define cublasCreate         cublasCreate_v2

    TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD需打开。
    """
    if cursor.kind == CursorKind.MACRO_DEFINITION:
        tk = list(cursor.get_tokens())
        if len(tk) == 2 and not tk[0].spelling.startswith('_') and tk[1].kind.name == 'IDENTIFIER':
            MACRO_MAP[tk[1].spelling] = tk[0].spelling


def actual_arg(cursor):
    """获取调用时传递的实参，忽略隐式类型转换/实例化。Cursor.kind应为CALL_EXPR。
    """
    root = cursor
    for _ in range(100):
        # CursorKind: MEMBER_REF_EXPR, DECL_REF_EXPR, TYPE_REF, STRING_LITERAL, INTEGER_LITERAL ...
        if 'REF' in cursor.kind.name or 'LITERAL' in cursor.kind.name:
            return cursor
        children = get_children(cursor)
        if children:
            cursor = children[0]
        else:
            break
    return root


def parent_stmt(cursor):
    """获取所属Statement对应的代码"""
    root = cursor
    for _ in range(100):
        # CursorKind: DECL_STMT, DEFAULT_STMT ...
        if cursor.kind.name.endswith('STMT'):
            return cursor
        parent = get_attr(cursor, 'parent')
        if parent:
            cursor = parent
        else:
            break
    return root


def parse_args(node):
    args = list()
    if node.kind == CursorKind.CALL_EXPR:
        parameters = [f'{x.type.spelling} {x.spelling}' for x in node.referenced.get_arguments()]
        arguments = list(node.get_arguments())
        # 构造函数调用时，get_arguments()获取不到实参，referenced.get_arguments()可以获取形参。
        if parameters and not arguments:
            ref_end = -1
            # 隐式调用（通常为构造函数调用）子节点均为参数，显式(构造函数)调用子节点包含命名空间+（类型）引用+参数。
            children = get_children(node)
            for i, child in enumerate(children):
                if child.kind == CursorKind.TYPE_REF:
                    ref_end = i
                    break
            arguments = children[ref_end + 1:]

        for param, x in zip(parameters, arguments):
            x = skip_implicit(x)
            if not x:  # 有默认值的Keyword参数，如果实参未传，则为None
                continue
            x = actual_arg(x)
            # 或直接读取代码：read_cursor(x)
            spelling = auto_match(x).spelling  # 参数通常未记录info，无法获取info.spelling
            if is_usr_code(get_attr(x, 'referenced.location.file.name')):
                start = get_attr(x, 'referenced.extent.start')
                src_loc = f"{get_attr(start, 'file.name')}, {get_attr(start, 'line')}:" \
                          f"{get_attr(start, 'column')}"
                src_code = read_cursor(x.referenced)
            else:
                src_loc = 'NO_REF'
                src_code = 'NO_REF'
            args.append(f'{param} | {spelling} | {src_code} | {src_loc}')
    return args


def parse_info(node, cwd=None):
    if node.kind == CursorKind.TRANSLATION_UNIT:
        file = node.spelling
    else:
        if not get_attr(node, 'location.file'):
            file = None
        else:
            file = os.path.normpath(node.location.file.name)

    macro_map(node)
    # 如果对于系统库直接返回None，可能会导致部分类型无法解析，但是解析系统库会导致性能下降。
    usr_code = is_usr_code(file)

    if usr_code:
        SCANNED_FILES.append(file)
        hit = False
        if not getattr(node, 'scanned', False):
            hit, (result_type, spelling, api, definition, source), cuda_en = filter_acc(node)
        hit = hit and not getattr(node, 'implicit', False)

        if hit:
            api = MACRO_MAP.get(api, api)
            loc = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
                  f"{get_attr(node, 'extent.start.column')}"
            args = parse_args(node)
            item = {
                'API': api,
                'CUDAEnable': cuda_en,
                'Location': loc,
                'Context(形参 | 实参 | 来源代码 | 来源位置)': args,
            }
            RESULTS.append(item)

    children = list()
    for c in get_children(node):
        c_info = parse_info(c, cwd)
        if c_info:
            children.append(c_info)

    if not usr_code:
        info = None
        return info
    location = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
               f"{get_attr(node, 'extent.start.column')}-{get_attr(node, 'extent.end.column')}"

    # node的属性和方法：kind.name/type.kind.name/get_usr()/displayname/spelling/type.spelling/hash
    # 其他可记录信息：get_attr(node, 'referenced.kind.name')/api/result_type/source/definition/get_ref_def(node)/children
    info = {
        'kind': node.kind.name,
        'type_kind': node.type.kind.name,
        'ref_kind': get_attr(node, 'referenced.kind.name'),
        'spelling': node.spelling,
        'type': node.type.spelling,
        'hash': node.hash,
        'location': location,
        'children': children
    }

    return info


class Parser:
    # creates the object, does the inital parse
    def __init__(self, path):
        logger.info(f'Scanning file: {path}')
        self.index = Index.create()  # 若为单例模型，是否有加速作用
        # args: '-Xclang', '-ast-dump', '-fsyntax-only', '-std=c++17', "-I/path/to/include"
        # option: TranslationUnit.PARSE_PRECOMPILED_PREAMBLE, TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        includes = [f'-I{x}' for x in KitConfig.INCLUDES.values() if x]
        self.tu = self.index.parse(path,
                                   args=includes,
                                   options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    def parse(self, log=False):
        global RESULTS, SCANNED_FILES, MACRO_MAP
        RESULTS.clear()
        MACRO_MAP.clear()

        for d in self.tu.diagnostics:
            logger.warning(f'Code diagnose：{get_diag_info(d)}')
            if d.severity > KitConfig.TOLERANCE:
                logger.warning(f'Diagnostic severity {d.severity} > tolerance {KitConfig.TOLERANCE}, skip this file.')
                return dict()

        cwd = os.path.dirname(self.tu.spelling)  # os.path.abspath(os.path.normpath(tu.spelling))
        start = time.time()
        info = parse_info(self.tu.cursor, cwd=cwd)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        if logger.level == logging.DEBUG:
            dump = self.tu.spelling.replace('/', '.')
            os.makedirs('temp/', exist_ok=True)
            IOUtil.json_safe_dump(info, f'temp/{dump}.json')
            logger.debug(f'Ast saved in：temp/{dump}.json')
        if log:
            logger.info(RESULTS)
        return RESULTS


if __name__ == '__main__':
    p = Parser('../examples/classify.cpp')
    p.parse(log=True)
