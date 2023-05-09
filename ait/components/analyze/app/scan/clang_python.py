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

"""
Possible methods:
1. libclang: https://github.com/JonathanPierce/PyTokenize
2. http://eli.thegreenplace.net/2011/07/03/parsing-c-in-python-with-clang/
3. pygments.lexers.c_cpp.CppLexer: http://pygments.org/docs/lexers/#lexers-for-c-c-languages
4. PyParsing
5. http://gccxml.github.io/HTML/Index.html

LLVM python bindings: https://github.com/llvm/llvm-project/tree/main/clang/bindings/python
libclang(mirrored LLVM python bindings): https://readthedocs.org/projects/libclang/
clang(mirrored LLVM python bindings): https://pypi.org/project/clang/
libclang case: https://eli.thegreenplace.net/2011/07/03/parsing-c-in-python-with-clang/

Note that the library is named libclang,
the package clang on PyPi is another package and doesn't bundle the prebuilt shared library.
"""

import os
import json
import re
import time
import linecache
from pprint import pprint

from clang.cindex import Index, LinkageKind, CursorKind, TypeKind, TranslationUnit, Config

from common.kit_config import KitConfig
from utils.log_util import logger
from utils.lib_util import get_sys_path

SYS_PATH = get_sys_path()
SCANNED_FILES = list()
RESULTS = list()
INCLUSION = 'inclusion'
# set the config
if not Config.loaded:
    Config.set_library_file(KitConfig.lib_clang_path)


def get_diag_info(diag):
    return {'severity': diag.severity,
            'location': diag.location,
            'spelling': diag.spelling,
            'ranges': diag.ranges,
            'fixits': diag.fixits}


def cuda_enabled(file, include, namespace=None):
    """判断该文件是否为加速库内cuda相关文件。"""
    if not isinstance(include, list):
        include = [include]
    for x in include:
        if x == 1 or x in file:
            return True
    return False


def in_acc_lib(file, cursor):
    """判断该文件是否为加速库文件。"""
    if not file:
        return False, False, ''
    for lib, v in KitConfig.acc_libs.items():
        if lib in file:
            if not v:
                cuda_en = False
                add_ns = ''
            else:
                cuda_en = cuda_enabled(file, v[1])
                add_ns = add_namespace(cursor, v[0])
            return True, cuda_en, add_ns
    return False, False, ''


def get_ref_def(cursor):
    ref = cursor.referenced
    decl = cursor.get_definition()
    result = list()
    if ref is not None:
        result.extend([f'ref: {ref.displayname}', f'{ref.location}'])  # {ref.extent.start.line}
    if decl is not None:
        result.extend([f'def: {decl.displayname}', f'{decl.location}'])
    return result


def get_attr(obj, attr=None, default=None):
    """"""
    attrs = attr.split('.')
    for a in attrs:
        obj = getattr(obj, a, default)
        if not obj:
            return obj
    return obj


def get_namespace(children):
    """例如cv::cudacodec::VideoReader"""
    namespace = ''
    for child in children:
        if child.kind in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF]:
            child.scanned = True
            namespace += child.spelling + '::'
        if child.kind == CursorKind.TYPE_REF:
            child.scanned = True
            if namespace in child.type.spelling:
                namespace = child.type.spelling + '::'
            else:
                namespace += child.type.spelling + '::'
    return namespace


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
    nsc = re.findall('(?:@N@\w+){1,1000}', usr[:index])
    nss = ['::'.join(x[3:].split('@N@')) for x in nsc]
    for namespace in namespaces:
        for ns in reversed(nss):
            if namespace in ns:  # namespace可能是pattern，不是完整namespace
                return ns
    return ''


def get_children(cursor):
    if not cursor:
        return list()
    if not hasattr(cursor, 'children'):
        cursor.children = list(cursor.get_children())  # 临时属性，便于特殊处理子节点
    for child in cursor.children:
        child.parent = cursor
        if hasattr(cursor, 'scanned'):
            child.scanned = cursor.scanned
    return cursor.children


def set_attr_children(cursor, attr, value):
    for child in get_children(cursor):
        setattr(child, attr, value)

def skip_cast(c):
    """通常是隐式类型转换或者指针。

    一个隐式转换应只有1个后继节点。
    一个指针应只有1个后继节点，c.type.kind == TypeKind.POINTER
    """
    if c.kind == CursorKind.UNEXPOSED_EXPR:
        children = get_children(c)
        if len(children):
            return skip_cast(children[0])
        else:
            return None
    return c


"""
复杂示例

示例1：examples/classify.cpp。如下代码无法通过referenced获取源头，CALL_EXPR属于TypeKind.DEPENDENT，即参数依赖get函数调用。
代码：
VideoCapture cap;
cap.open(parser.get<String>("input"));

字段：spelling, CursorKind, type, TypeKind
{'', CALL_EXPR, <dependent type>, DEPENDENT}
    {'', MEMBER_REF_EXPR, <bound member function type>, UNEXPOSED}
        {'cap', DECL_REF_EXPR, cv::VideoCapture, RECORD}
        {'open', OVERLOADED_DECL_REF, '', INVALID}
    {'', UNEXPOSED_EXPR, <dependent type>, DEPENDENT}
        {'', MEMBER_REF_EXPR, <bound member function type>, UNEXPOSED}
            {'parser', DECL_REF_EXPR, cv::CommandLineParser, RECORD}
            {'get', OVERLOADED_DECL_REF, '', INVALID}
            {'cv::String', TYPE_REF, cv::String, TYPEDEF}
        {'\"input\"', STRING_LITERAL, const char[6], CONSTANTARRAY}

示例2：video_reader.cpp
代码：cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

字段：spelling, type, CursorKind, TypeKind
{"d_reader", cv::Ptr<cv::cudacodec::VideoReader>, VAR_DECL, ELABORATED}
    {"cv", '', NAMESPACE_REF, INVALID}
    {"Ptr", '', TEMPLATE_REF, INVALID}
    {"cv", '', NAMESPACE_REF, INVALID}
    {"cudacodec", '', NAMESPACE_REF, INVALID}
    {"class cv::cudacodec::VideoReader", 'cv::cudacodec::VideoReader', TYPE_REF, RECORD}
    # = 右侧
    {'', 'cv::Ptr<cv::cudacodec::VideoReader>', UNEXPOSED_EXPR, ELABORATED}
        {'', 'cv::Ptr<cv::cudacodec::VideoReader>', CALL_EXPR, ELABORATED}  # 构造函数调用
            {'', "Ptr<cv::cudacodec::VideoReader>", UNEXPOSED_EXPR, UNEXPOSED}
                {'', "Ptr<cv::cudacodec::VideoReader>", UNEXPOSED_EXPR, UNEXPOSED}
                    {'createVideoReader', "Ptr<cv::cudacodec::VideoReader>", CALL_EXPR, UNEXPOSED}  # 函数调用
                        {'createVideoReader', "Ptr<cv::cudacodec::VideoReader> (*)(const cv::String &, const bool)", UNEXPOSED_EXPR, POINTER}
                            {'createVideoReader', "Ptr<cv::cudacodec::VideoReader> (*)(const cv::String &, const bool)", DECL_REF_EXPR, FUNCTIONPROTO}
                                {'cv', '', NAMESPACE_REF, INVALID}
                                {"cudacodec", '', NAMESPACE_REF, INVALID}
                        {"fname", "const std::string", DECL_REF_EXPR, ELABORATED}

示例3：hog.cpp
代码：Mat detector = gpu_hog->getDefaultPeopleDetector();

字段：spelling, type, CursorKind, TypeKind
{"detector", "cv::Mat", "VAR_DECL", "RECORD"}
    {"class cv::Mat", "cv::Mat", "TYPE_REF", "RECORD"}
    # = 右侧
    {"", "cv::Mat", "UNEXPOSED_EXPR", "RECORD"}
        # cv::Mat()构造函数调用
        {"", "cv::Mat", "CALL_EXPR", "RECORD"}
            {"", "cv::Mat", "UNEXPOSED_EXPR", "RECORD"}
                {"", "cv::Mat", "UNEXPOSED_EXPR", "RECORD"}
                    # getDefaultPeopleDetector()方法调用
                    {"getDefaultPeopleDetector", "cv::Mat", "CALL_EXPR", "RECORD"}
                        {"getDefaultPeopleDetector", "<bound member function type>", "MEMBER_REF_EXPR", "UNEXPOSED"}
                            # 方法所属对象
                            {"operator->", "const cv::cuda::HOG *", "UNEXPOSED_EXPR", "POINTER"}
                                # ->内部的get()调用
                                {"operator->", "cv::cuda::HOG *", "CALL_EXPR", "POINTER"}
                                    {"gpu_hog", "const cv::Ptr<cv::cuda::HOG>", "UNEXPOSED_EXPR", "RECORD"}
                                        {"gpu_hog", "cv::Ptr<cv::cuda::HOG>", "DECL_REF_EXPR", "ELABORATED"}
                                    # ->函数引用
                                    {"operator->", "cv::cuda::HOG *(*)() const noexcept", "UNEXPOSED_EXPR", "POINTER"}
                                        {"operator->", "cv::cuda::HOG *() const noexcept", "DECL_REF_EXPR", "FUNCTIONPROTO"}
                            # 无入参

示例4：example.cpp
代码：
Mat image;
VideoCapture capture;
capture >> image;

字段：spelling, type, CursorKind, TypeKind
{"operator>>", "cv::VideoCapture", "CALL_EXPR", "RECORD"}
    {"capture", "cv::VideoCapture", "DECL_REF_EXPR", "RECORD"}
    {"operator>>", "cv::VideoCapture &(*)(cv::Mat &)", "UNEXPOSED_EXPR", "POINTER"}
        {"operator>>", "cv::VideoCapture &(cv::Mat &)", "DECL_REF_EXPR", "FUNCTIONPROTO"}
    {"image", "cv::Mat", "DECL_REF_EXPR", "RECORD"}
"""


# 可作为最小呈现粒度的Statement：
# CursorKind.UNEXPOSED_STMT
# # Adaptor class for mixing declarations with statements and expressions.
# CursorKind.DECL_STMT

def default(c):
    if c.referenced:
        definition = c.referenced.displayname
        source = get_attr(c.referenced, 'location.file.name')
    else:
        definition = c.displayname
        source = get_attr(c, 'location.file.name')
    return c.type.spelling, c.spelling, c.spelling, definition, source


def var_decl(c):
    """变量声明，不包括赋值。

    通常TypeKind为：TYPEDEF，RECORD
    extent包含类型和变量名，无需修改。
    如果非原生类型，通常后继节点有命名空间NAMESPACE_REF（可选），类型引用TYPE_REF，构造函数CALL_EXPR（可选），可通过referenced获取源头。
    示例1：vector<vector<cv::Point> >

    Returns:
        type_extension, spelling, api, definition, source_location
    """
    children = get_children(c)
    api = c.type.spelling
    if children:  # TYPEDEF RECORD
        for i, child in enumerate(children):  # 示例2
            if child.kind in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF]:
                child.scanned = True  # 用于标识是否已被扫描
            if child.kind == CursorKind.TYPE_REF:
                definition = get_attr(child, 'referenced.displayname')
                source = get_attr(child, 'referenced.location.file.name')
                if i + 1 < len(children) and children[i + 1].kind == CursorKind.CALL_EXPR:
                    definition = call_expr(children[i + 1])[3]
                break
        else:  # 原生类型变量声明 = 加速库调用
            return default(c)
    else:  # 原生类型变量声明
        source = get_attr(c, 'location.file.name')
        definition = c.get_definition().displayname
    # TODO(dyh)：是否去除api的namespace
    return api, f'{c.type.spelling} {c.spelling}', api, definition, source


def parm_decl(c):
    """参数声明，不包括赋值。

    extent包含类型和变量名，无需修改。
    如果非原生类型，后继节点有命名空间NAMESPACE_REF（可选），类型引用TYPE_REF，可通过referenced获取源头。
    Returns:
        type_extension, spelling, api, definition, source_location
    """
    children = get_children(c)
    # c.children = children  # 临时属性，用于特殊处理子节点
    if children:
        for i, child in enumerate(children):
            if child.kind in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF]:
                child.scanned = True  # 用于标识是否已被扫描
            if child.kind == CursorKind.TYPE_REF:
                definition = get_attr(child, 'referenced.displayname')
                source = get_attr(child, 'referenced.location.file.name')
                if i + 1 < len(children) and children[i + 1].kind == CursorKind.CALL_EXPR:
                    definition = call_expr(children[i + 1])[3]
                break
        else:  # 原生类型变量声明 = 加速库调用
            return default(c)
    else:  # 原生类型变量声明
        source = get_attr(c, 'location.file.name')
        definition = c.get_definition().displayname
    if c.type.kind in (TypeKind.RVALUEREFERENCE, TypeKind.LVALUEREFERENCE):
        api = c.type.get_pointee().spelling
    else:
        api = c.type.spelling
    return api, f'{c.type.spelling} {c.spelling}', api, definition, source


def call_expr(c):
    """
    重载运算：operator=, a=b
        1. 对象引用：DECL_REF_EXPR
        2. 函数引用：DECL_REF_EXPR
        3. 参数

    重载运算：std::cout << "..." << "..." << std::endl;
    1. CALL_EXPR：operator<<
        1. CALL_EXPR：operator<<
            1. CALL_EXPR：operator<<
                1. DECL_REF_EXPR of TYPEDEF：cout
                2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<
                3. 参数
            2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<，将1的结果作为类型引用
            3. 参数
        2. DECL_REF_EXPR of FUNCTIONPROTO: operator<<，将1的结果作为类型引用
        3. DECL_REF_EXPR of FUNCTIONPROTO: endl，将2的结果作为类型引用

    重载运算：cv::Size imgSize = img.size();
    1. CALL_EXPR：operator()
        1. MEMBER_REF_EXPR of RECORD：size
            1. DECL_REF_EXPR of ELABORATED: img
        2. DECL_REF_EXPR of FUNCTIONPROTO: operator()

    函数调用：a()
        1. 函数引用：DECL_REF_EXPR
        2. 参数

    方法调用：a.b()
        1. 方法引用：MEMBER_REF_EXPR
            1. 对象引用：DECL_REF_EXPR
        2. 参数

    注意存在隐式类调用cv::Mat m，此时type为cv::Mat，definition为Mat，没有后继节点。
    extent包含命名空间、函数名、()，无需修改。
    通常CALL_EXPR可通过referenced获取源头，但示例1不适用。
    Returns:
        type_extension, spelling, api, definition, source_location
    """
    c.implicit = True
    children = get_children(c)
    if not children:
        return default(c)
    c0 = skip_cast(children[0])
    if not c0:
        return default(c)

    op_overload = 'operator' in c.spelling  # TODO(dyh)：需要判断的更加准确
    for i, child in enumerate(children):
        child = skip_cast(child)
        if not child:
            continue
        if child.kind in [CursorKind.NAMESPACE_REF, CursorKind.TEMPLATE_REF]:
            child.scanned = True  # 用于标识是否已被扫描
        if op_overload and c.spelling == child.spelling:
            if i > 0:
                child.scanned = True  # 用于标识是否已被扫描

    type_x, spelling, api, definition, source =  default(c)
    if op_overload:
        c.implicit = False
        if c0.kind.name in whole_dict:
            name, _, attr, _, _ = whole_dict.get(c0.kind.name)(c0)
        else:
            name, _, attr, _, _ = default(c0)
        if c0.kind == CursorKind.MEMBER_REF_EXPR:
            # 运算符重载，且最近子节点为MEMBER_REF_EXPR，则必为属性引用，而非方法引用，否则最近子节点为CALL_EXPR。
            api = f"{attr}{c.spelling[8:]}"  # 例如呈现cv::Mat.size()，实际为cv::MatSize()
            c0.scanned = True
        else:
            api = f"{name}{c.spelling[8:]}"  # 例如呈现cv::FileStorage[]
            if c0.kind == CursorKind.DECL_REF_EXPR:
                c0.scanned = True
    return type_x, spelling, api, definition, source


def member_ref_expr(c):
    """
    extent包含对象名、方法名，不包括()、参数，需修改。

    - 通常有后继节点：对象名DECL_REF_EXPR，函数名OVERLOADED_DECL_REF（可选），可通过referenced获取源头，但示例1不适用。
    - 若为类属性，则无后继节点。

    Returns:
        type_extension, spelling, api, definition, source_location
    """
    api = c.spelling  # 示例1不适用
    if c.referenced:
        type_x = c.referenced.result_type.spelling  # 方法返回值类型
        definition = c.referenced.displayname
        source = get_attr(c.referenced, 'location.file.name')
    else:
        type_x = None
        definition = None
        source = None
    children = get_children(c)
    if not children:
        return default(c)
    c0 = skip_cast(children[0])
    if not c0:
        return default(c)
    if c0.kind == CursorKind.DECL_REF_EXPR:
        c0.scanned = True
        cls, obj, _, _, _ = decl_ref_expr(c0)  # 对象名的source在声明/实例化对象的地方，不在加速库里
    else:
        if c0.kind.name not in whole_dict:
            cls, obj, _, _, _ = default(c0)
        else:
            cls, obj, _, _, _ = whole_dict[c0.kind.name](c0)

    cls = cls.replace('const ', '')  # 去除const
    return type_x, f'{obj}.{api}', f'{cls}.{api}', definition, source


def decl_ref_expr(c):
    """
    Returns:
        type_extension, spelling, api, definition, source_location
    """
    # 若为函数引用，需从children[0]的namespace获取
    # extent包含对象名、方法名，不包括()、参数，需修改。
    # type为<overloaded function type>，referenced
    definition = c.referenced.displayname if c.referenced else None
    if c.type.kind == TypeKind.OVERLOAD:
        children = get_children(c)
        if len(children) < 1:
            raise RuntimeError(f'DECL_REF_EXPR of Typekind OVERLOAD {c.spelling} {c.location} 应有后继节点：' \
                                   f'命名空间NAMESPACE_REF（可选） 函数名OVERLOADED_DECL_REF')
        spelling = get_namespace(children) + children[-1].spelling
        api = spelling
        type_x = get_attr(children[-1], 'referenced.result_type.spelling')  # 函数返回值类型
        source = get_attr(children[0], 'referenced.location.file.name')
    # extent包含对象名、方法名，不包括()、参数，需修改。
    elif c.type.kind == TypeKind.FUNCTIONPROTO:
        children = get_children(c)
        # DECL_REF_EXPR of Typekind FUNCTIONPROTO 有后继节点：命名空间NAMESPACE_REF（可选），类型TYPE_REF（可选）
        spelling = get_namespace(children) + c.spelling
        api = spelling
        type_x = get_attr(c, 'referenced.result_type.spelling')  # 函数返回值类型
        source = get_attr(c, 'referenced.location.file.name')
    # 若为对象引用，可从type获取类型归属，ELABORATED/RECORD/CONSTANTARRAY/TYPEDEF/ENUM
    # extent包含对象名，无需修改。
    elif c.type.kind == TypeKind.ENUM:
        type_x, spelling, api, definition, source = default(c)
        if 'anonymous enum' not in c.type.spelling:
            api = f'{c.type.spelling}.{api}'
        set_attr_children(c, 'scanned', True)
    else:
        return default(c)

    return type_x, spelling, api, definition, source


def overloaded_decl_ref(c):
    return default(c)


def array_subscript_expr(c):
    """
    对应TypeKind.CONSTANTARRAY

    1. ARRAY_SUBSCRIPT_EXPR：数组索引表达式
        1. 嵌套对象/DECL_REF_EXPR/MEMBER_REF_EXPR
        2. 索引

    Args:
        c:
        type_src: 获取数组类型的source。

    Returns:
        type_extension, spelling, api, definition, source_location
    """
    return default(c)


def template_ref(c):
    return default(c)


def type_ref(c):
    """
    Args:
        c: cursor.

    Returns:
        type_extension, spelling, api, definition, source_location
    """
    definition = get_attr(c, 'referenced.displayname')
    source = get_attr(c, 'referenced.location.file.name')
    return c.type.spelling, c.spelling, c.type.spelling, definition, source


def namespace_ref(c):
    """
    Args:
        c: cursor.

    Returns:
        type_extension, spelling, api, definition, source_location
    """
    return None, c.spelling, None, None, get_attr(c, 'referenced.location.file.name')


def inclusion_directive(c):
    prefix = '' if c.spelling.startswith('/') else '/'
    return None, c.spelling, None, None, prefix + c.spelling  # c.get_included_file().name会遇到Assert错误


# 小粒度分析CursorKind
small_dict = {
    'MEMBER_REF_EXPR': member_ref_expr, 'DECL_REF_EXPR': decl_ref_expr, 'OVERLOADED_DECL_REF': overloaded_decl_ref,
    'TYPE_REF': type_ref, 'TEMPLATE_REF': template_ref, 'NAMESPACE_REF': namespace_ref,
    'INCLUSION_DIRECTIVE': inclusion_directive,
    'CALL_EXPR': call_expr
}

# 大粒度分析CursorKind
large_dict = {
    'VAR_DECL': var_decl, 'PARM_DECL': parm_decl,
}

# 工具性分析CursorKind
other_dict = {
    'ARRAY_SUBSCRIPT_EXPR': array_subscript_expr
}

whole_dict = {**small_dict, **large_dict, **other_dict}

filter_dict = small_dict
helper_dict = large_dict  # 用于提前对节点进行处理，比如VAR_DECL的命名空间、FUNCTIONPROTO的参数等
if KitConfig.level == 'large':
    filter_dict = whole_dict
    helper_dict = dict()


def matcher(c):
    if c.kind.name in whole_dict:
        return whole_dict.get(c.kind.name)(c)
    else:
        return default(c)


def filter_acc(cursor):
    hit = False
    type_x = None
    spelling = None
    api = None
    definition = None
    source = None
    cuda_en = False
    ns = ''

    if cursor.kind.name in helper_dict:  # 用于提前对节点进行处理，比如VAR_DECL的命名空间、FUNCTIONPROTO的参数等
        type_x, spelling, api, definition, source = helper_dict[cursor.kind.name](cursor)
    if cursor.kind.name in filter_dict:
        type_x, spelling, api, definition, source = filter_dict[cursor.kind.name](cursor)
        hit, cuda_en, ns = in_acc_lib(source, cursor)

    if ns and api and f'{ns}::' not in api:
        core = api.split('::')[-1]
        api = f'{ns}::{core}'
    return hit, type_x, spelling, api, definition, source, cuda_en


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


def read_code(file_path, start, end):
    with open(file_path, 'r') as f:
        f.seek(start)
        return f.read(end - start)


def is_usr_code(file):
    usr_code = True
    if not file or any(file.startswith(p) for p in SYS_PATH):
        usr_code = False
    return usr_code


def get_info(node, cwd=None):
    if node.kind == CursorKind.TRANSLATION_UNIT:
        file = node.spelling
    else:
        if not get_attr(node, 'location.file'):
            file = None
        else:
            file = os.path.normpath(node.location.file.name)

    # 如果对于系统库直接返回None，可能会导致部分类型无法解析，但是解析系统库会导致性能下降。
    usr_code = is_usr_code(file)

    hit = False
    if usr_code:
        SCANNED_FILES.append(file)
        if not getattr(node, 'scanned', False):
            hit, type_x, spelling, api, definition, source, cuda_en = filter_acc(node)
        hit = hit and not getattr(node, 'implicit', False)

    children = list()
    for c in get_children(node):
        c_info = get_info(c, cwd)
        if c_info:
            children.append(c_info)

    if not usr_code:
        return None

    location = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
               f"{get_attr(node, 'extent.start.column')}-{get_attr(node, 'extent.end.column')}"
    args = list()
    for x in node.get_arguments():
        x = skip_cast(x)
        if x:
            args.append(f"{x.spelling}: {get_attr(x, 'referenced.hash')}")
        else:
            args.append(":")
    info = {
        'kind': node.kind.name,
        'type_kind': node.type.kind.name,
        'displayname': node.displayname,
        'spelling': node.spelling,
        'type': node.type.spelling,
        'hash': node.hash,
        'args': args,
        'location': location,
        'ref_def': get_ref_def(node),
        'children': children
    }

    if hit:
        loc = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
              f"{get_attr(node, 'extent.start.column')}"
        item = {
            'api': api,
            'cuda_en': cuda_en,
            'location': loc,
            # 'args': args
        }
        RESULTS.append(item)
    return info


class Parser:
    # creates the object, does the inital parse
    def __init__(self, path):
        logger.info(f'Scanning file: {path}')
        self.index = Index.create()  # TODO(dyh):若为单例模型，是否有加速作用
        # args: ['-Xclang', '-ast-dump', '-fsyntax-only', '-std=c++17', "-I/path/to/include"]
        # TranslationUnit.PARSE_PRECOMPILED_PREAMBLE, TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        self.tu = self.index.parse(path,
                                   args=[f'-I{KitConfig.opencv_include_path}'],
                                   options=TranslationUnit.PARSE_NONE)

    def parse(self, log=False):
        global RESULTS, SCANNED_FILES
        RESULTS.clear()
        diag_info = [get_diag_info(d) for d in self.tu.diagnostics]
        if diag_info:
            logger.info(f'Code diagnose：{diag_info}')
        cwd = os.path.dirname(self.tu.spelling)  # os.path.abspath(os.path.normpath(tu.spelling))
        start = time.time()
        info = get_info(self.tu.cursor, cwd=cwd)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        dump = self.tu.spelling.replace('/', '.')
        os.makedirs('temp/', exist_ok=True)
        with open(f'temp/{dump}.json', 'w') as f:
            json.dump(info, f, indent=4)
        logger.debug(f'语法树临时文件已保存到：temp/{dump}.json')
        if log:
            pprint(RESULTS)
        return RESULTS

    def filter(self):
        pass


if __name__ == '__main__':
    p = Parser('../examples/classify.cpp')
    p.parse(log=True)
