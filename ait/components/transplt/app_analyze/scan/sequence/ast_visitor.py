import re

from clang.cindex import CursorKind
from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.sequence.seq_utils import FuncAttr


def _in_acc_lib(file, cursor):
    """判断该文件是否为加速库文件。"""
    if not file:
        return False, False, ''
    for lib, v in KitConfig.ACC_LIBS.items():
        if lib in file:  # 待ACC_LIBS的Pattern改为全路径后，可以使用file.startswith(lib)
            if not v:
                cuda_en = False
                usr_ns = ''
            else:
                # get relative path
                new_file = file if not file.startswith(lib) else file.replace(lib, '')
                cuda_en = _cuda_enabled(new_file, v[1])
                usr_ns = _usr_namespace(cursor, v[0])
                cursor.lib = v[3]
            return True, cuda_en, usr_ns
    return False, False, ''


def _cuda_enabled(file, include, namespace=None):
    """判断该文件是否为加速库内cuda相关文件。"""
    if not isinstance(include, list):
        include = [include]

    for x in include:
        if x == '':
            continue

        if x == 1 or x in file:
            return True
    return False


def _usr_namespace(cursor, namespaces):
    """解析get_usr中的命名空间。

    例如：
    "c:@N@cv@ST>1#T@Ptr"：cv::Ptr<T>
    "c:@N@cv@E@WindowFlags@WINDOW_OPENGL"：cv::WindowFlags.WINDOW_OPENGL
    "c:@N@cv@N@cuda@S@GpuMat@F@GpuMat#*$@N@cv@N@cuda@S@GpuMat@S@Allocator#"：cv::cuda::CpuMat::GptMat
    "c:@N@cv@S@Ptr>#$@N@cv@N@cudacodec@S@VideoReader@F@operator->#1"：cv::Ptr<cv::cudacodec::VideoReader>
    """
    if not namespaces or not cursor.referenced:
        return ''
    if not isinstance(namespaces, list):
        namespaces = [namespaces]
    usr = cursor.referenced.get_usr()
    index = usr.find(cursor.referenced.spelling)  # 忽略"@S@GpuMat@F@GpuMat"这种重复的影响
    if index == -1:
        return ''
    nsc = re.findall(r'(?:@N@\w+){1,1000}', usr[:index])
    nss = ['::'.join(x[3:].split('@N@')) for x in nsc]
    for namespace in namespaces:
        for ns in nss:
            if namespace in ns:  # namespace可能是pattern，不是完整namespace
                return ns
    return ''


def _visit_call_expr(node):
    print()


def _visit_function_decl(node):
    func_attr = FuncAttr()
    func_attr.func_name = node.spelling
    func_attr.return_type = node.result_type.spelling
    func_attr.location = node.location
    func_attr.hash_code = node.hash

    c = node.semantic_parent
    while c and c.kind != CursorKind.TRANSLATION_UNIT:
        func_attr.namespace = c.spelling + '::' + func_attr.namespace
        if not c:
            break
        c = c.semantic_parent
    return func_attr


def visit(node, sub_rst, result):
    hit = False
    cursor_kind = node.kind
    if cursor_kind == CursorKind.FUNCTION_DECL:
        if sub_rst:
            val = []
            val.extend(sub_rst)
            result.append(val)
            sub_rst.clear()
        func_attr = _visit_function_decl(node)
        sub_rst.append(func_attr)
        hit = True
    elif cursor_kind == CursorKind.CALL_EXPR:
        ref_kind = node.referenced.kind.name
        if ref_kind in ['CXX_METHOD', 'FUNCTION_DECL']:
            func_attr = _visit_function_decl(node)
            hit, cuda_en, ns = _in_acc_lib(node.referenced.location.file.name, node)
            if hit:
                sub_rst.append(func_attr)

    return hit, FuncAttr()
