import os
import time

from app_analyze.scan.clang_parser import *
from app_analyze.scan.clang_utils import *
from app_analyze.scan.sequence.seq_utils import FuncAttr, ObjInfo
from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.log_util import logger


def _get_api_type(file, cursor):
    """判断该文件是否为加速库文件。"""
    arg_dict = {'cuda_en': False, 'usr_ns': '', 'api_type': 'invalid'}
    if not file:
        return arg_dict

    for lib, v in KitConfig.ACC_LIBS.items():
        if lib in file:  # 待ACC_LIBS的Pattern改为全路径后，可以使用file.startswith(lib)
            if v:
                # get relative path
                new_file = file if not file.startswith(lib) else file.replace(lib, '')
                arg_dict['cuda_en'] = cuda_enabled(new_file, v[1])
                arg_dict['usr_ns'] = usr_namespace(cursor, v[0])
                arg_dict['api_type'] = 'acc_lib'
                cursor.lib = v[3]
            return arg_dict

    if file.startswith(KitConfig.SOURCE_DIRECTORY):
        arg_dict['api_type'] = 'usr_defined'

    return arg_dict


def _get_input_args(node):
    args = list()

    refs = node.referenced
    if not refs:
        return args

    parameters = [f'{x.type.spelling} {x.spelling}' for x in node.referenced.get_arguments()]
    arguments = list(node.get_arguments())

    for param, x in zip(parameters, arguments):
        x = skip_implicit(x)
        if not x:  # 有默认值的Keyword参数，如果实参未传，则为None
            continue
        args.append(f'{x.type.spelling}')
    return args


def _get_obj_info(node, func_attr):
    info = call_expr(node)
    obj = ObjInfo()
    if info.api.endswith('.' + info.spelling):
        obj.cxx_record_name = info.api.replace('.' + info.spelling, '')
    elif info.api.endswith('->' + info.spelling):
        obj.cxx_record_name = info.api.replace('->' + info.spelling, '')
    elif info.api.endswith('::' + info.spelling):
        obj.cxx_record_name = info.api.replace('::' + info.spelling, '')
    elif info.api == info.spelling:
        obj.cxx_record_name = info.api
    else:
        raise Exception('Error annotation!')
    func_attr.obj_info = obj
    if func_attr.return_type == '':
        func_attr.return_type = info.result_type


def _visit_function_decl(node, api_type='invalid'):
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
    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)

    func_attr.is_usr_def = True if api_type == 'usr_defined' else False
    if api_type == 'acc_lib':
        func_attr.acc_name = get_attr(node, 'lib')
    return func_attr


def _visit_cxx_method(node, api_type='invalid'):
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
    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)

    _get_obj_info(node, func_attr)
    func_attr.is_cxx_method = True
    func_attr.is_usr_def = True if api_type == 'usr_defined' else False
    if api_type == 'acc_lib':
        func_attr.acc_name = get_attr(node, 'lib')

    return func_attr


def visit(node, sub_rst, result):
    if node.spelling in ['operator=', 'operator>>', 'operator[]', 'operator()']:
        return

    cursor_kind = node.kind
    if cursor_kind == CursorKind.FUNCTION_DECL:
        if sub_rst:
            val = []
            val.extend(sub_rst)
            result.append(val)
            sub_rst.clear()
        func_attr = _visit_function_decl(node, 'usr_defined')
        sub_rst.append(func_attr)
    elif cursor_kind == CursorKind.CALL_EXPR:
        if not node.referenced:
            return

        ref_kind = node.referenced.kind.name
        if ref_kind in ['CXX_METHOD', 'FUNCTION_DECL']:
            arg_dict = _get_api_type(node.referenced.location.file.name, node)
            api_type = arg_dict['api_type']
            if api_type != 'invalid':
                if ref_kind == 'CXX_METHOD':
                    func_attr = _visit_cxx_method(node, api_type)
                else:
                    func_attr = _visit_function_decl(node, api_type)

                sub_rst.append(func_attr)
    elif cursor_kind in [CursorKind.CONSTRUCTOR, CursorKind.CXX_METHOD]:
        if not node.referenced:
            return
        if sub_rst:
            val = []
            val.extend(sub_rst)
            result.append(val)
            sub_rst.clear()

        ref_kind = node.referenced.kind.name
        if ref_kind == cursor_kind.name:
            arg_dict = _get_api_type(node.referenced.location.file.name, node)
            api_type = arg_dict['api_type']
            if api_type == 'usr_defined':
                func_attr = _visit_cxx_method(node, api_type)
                sub_rst.append(func_attr)


class FuncParser(Parser):
    def __init__(self, path):
        super().__init__(path)

    def _parse_api(self, node, sub_rst, result):
        file = None
        if node.kind == CursorKind.TRANSLATION_UNIT:
            file = node.spelling
        else:
            if get_attr(node, 'location.file'):
                file = os.path.normpath(node.location.file.name)

        macro_map(node, file)
        typedef_map(node, file)

        usr_code = is_user_code(file)
        if usr_code:
            if not getattr(node, 'scanned', False):
                visit(node, sub_rst, result)

        children = list()
        for c in get_children(node):
            c_info = self._parse_api(c, sub_rst, result)
            if c_info:
                children.append(c_info)

        info = None
        if usr_code:
            info = node_debug_string(node, children)

        return info

    def _handle_call_seqs(self):
        pass

    def parse(self):
        for d in self.tu.diagnostics:
            if d.severity > KitConfig.TOLERANCE:
                logger.warning(f'Diagnostic severity {d.severity} > tolerance {KitConfig.TOLERANCE}, skip this file.')
                return dict()

        sub_rst = []
        result = []

        start = time.time()
        info = self._parse_api(self.tu.cursor, sub_rst, result)
        if sub_rst:
            result.append(sub_rst)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')

        # dump = self.tu.spelling.replace('/', '.')
        # os.makedirs('temp/', exist_ok=True)
        # IOUtil.json_safe_dump(info, f'temp/{dump}.json')
        # logger.debug(f'Ast saved in：temp/{dump}.json')

        return RESULTS
