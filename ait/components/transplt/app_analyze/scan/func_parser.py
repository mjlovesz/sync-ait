import os
import time

from app_analyze.scan.clang_parser import *
from app_analyze.scan.clang_utils import *
from app_analyze.scan.sequence.seq_desc import FuncDesc, ObjDesc, SeqDesc
from app_analyze.scan.sequence.seq_handler import SeqHandler
from app_analyze.scan.sequence.seq_utils import is_unused_api, save_api_seq, sort_apis
from app_analyze.scan.sequence.api_filter import *
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

    parameters = [f'{x.type.spelling}' for x in node.referenced.get_arguments()]
    arguments = list(node.get_arguments())

    for param, x in zip(parameters, arguments):
        x = skip_implicit(x)
        if not x:  # 有默认值的Keyword参数，如果实参未传，则为None
            continue
        args.append(param)
    return args


def _get_obj_info(node, func_attr):
    info = call_expr(node)
    obj = ObjDesc()
    if info.api.endswith('.' + info.spelling):
        obj.record_name = info.api.replace('.' + info.spelling, '')
    elif info.api.endswith('->' + info.spelling):
        obj.record_name = info.api.replace('->' + info.spelling, '')
    elif info.api.endswith('::' + info.spelling):
        obj.record_name = info.api.replace('::' + info.spelling, '')
    elif info.api == info.spelling:
        obj.record_name = info.api
    else:
        raise Exception('Error annotation!')
    func_attr.obj_info = obj
    if func_attr.return_type == '':
        func_attr.return_type = info.result_type


def _visit_function_decl(node, api_type='invalid'):
    func_attr = FuncDesc()
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

    # if func_attr.func_name == "drawText":
    #     print()
    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)

    func_attr.is_usr_def = True if api_type == 'usr_defined' else False
    if api_type == 'acc_lib':
        func_attr.acc_name = get_attr(node, 'lib')
    return func_attr


def _visit_cxx_method(node, api_type='invalid'):
    func_attr = FuncDesc()
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


def _visit_call_expr(node, rst, pth):
    for c in get_children(node):
        cursor_kind = c.kind

        func_attr = None
        if cursor_kind == CursorKind.CALL_EXPR:
            if not c.referenced:
                return

            ref_kind = c.referenced.kind.name
            if ref_kind in ['CXX_METHOD', 'FUNCTION_DECL']:
                arg_dict = _get_api_type(c.referenced.location.file.name, c)
                api_type = arg_dict['api_type']
                if api_type != 'invalid':
                    if ref_kind == 'CXX_METHOD':
                        func_attr = _visit_cxx_method(c, api_type)
                    else:
                        func_attr = _visit_function_decl(c, api_type)

                    if is_unused_api(func_attr):
                        func_attr = None

        if func_attr:
            cur_path = []
            cur_path.extend(pth),
            cur_path.append(c)
            rst.append((func_attr, cur_path))
        else:
            cur_path = pth
        _visit_call_expr(c, rst, cur_path)


def visit(node, seq_desc, result):
    skip_flag = False
    if node.spelling in GLOBAL_FILTER:
        return

    cursor_kind = node.kind
    if cursor_kind == CursorKind.FUNCTION_DECL:
        save_api_seq(seq_desc, result)
        func_attr = _visit_function_decl(node, 'usr_defined')
        seq_desc.api_seq.append(func_attr)
    elif cursor_kind in [CursorKind.CONSTRUCTOR, CursorKind.CXX_METHOD]:
        if not node.referenced:
            return

        save_api_seq(seq_desc, result)
        ref_kind = node.referenced.kind.name
        if ref_kind == cursor_kind.name:
            arg_dict = _get_api_type(node.referenced.location.file.name, node)
            api_type = arg_dict['api_type']
            if api_type == 'usr_defined':
                func_attr = _visit_cxx_method(node, api_type)
                seq_desc.api_seq.append(func_attr)
                seq_desc.has_usr_def = True

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

                if is_unused_api(func_attr):
                    return skip_flag

                if api_type == 'usr_defined':
                    seq_desc.has_usr_def = True

                skip_flag = True
                rst = [(func_attr, [node])]
                pth = [node]
                _visit_call_expr(node, rst, pth)

                rst_size = len(rst)
                if rst_size == 1:
                    seq_desc.api_seq.append(rst[0][0])
                else:
                    sort_apis(rst)
                    seq_desc.api_seq.extend([val[0] for val in rst])

    return skip_flag


class FuncParser(Parser):
    def __init__(self, path):
        super().__init__(path)

    def _parse_api(self, node, seq_desc, result):
        file = None
        if node.kind == CursorKind.TRANSLATION_UNIT:
            file = node.spelling
        else:
            if get_attr(node, 'location.file'):
                file = os.path.normpath(node.location.file.name)

        macro_map(node, file)
        typedef_map(node, file)

        skip_flag = False
        usr_code = is_user_code(file)
        if usr_code and not getattr(node, 'scanned', False):
            skip_flag = visit(node, seq_desc, result)

        children = list()
        if skip_flag:
            info = node_debug_string(node, children)
        else:
            for c in get_children(node):
                c_info = self._parse_api(c, seq_desc, result)
                if c_info:
                    children.append(c_info)

            info = None
            if usr_code:
                info = node_debug_string(node, children)

        return info

    @staticmethod
    def _handle_call_seqs(seqs):
        SeqHandler.union_api_seqs(seqs)
        print()

    def parse(self):
        for d in self.tu.diagnostics:
            if d.severity > KitConfig.TOLERANCE:
                logger.warning(f'Diagnostic severity {d.severity} > tolerance {KitConfig.TOLERANCE}, skip this file.')
                return dict()

        seq_desc = SeqDesc()
        result = []

        start = time.time()
        info = self._parse_api(self.tu.cursor, seq_desc, result)
        save_api_seq(seq_desc, result)
        logger.debug(f'Time elapsed： {time.time() - start:.3f}s')
        self._handle_call_seqs(result)

        # dump = self.tu.spelling.replace('/', '.')
        # os.makedirs('temp/', exist_ok=True)
        # IOUtil.json_safe_dump(info, f'temp/{dump}.json')
        # logger.debug(f'Ast saved in：temp/{dump}.json')

        return RESULTS
