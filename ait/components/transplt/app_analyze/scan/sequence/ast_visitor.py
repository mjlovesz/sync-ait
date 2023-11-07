import re

from clang.cindex import CursorKind
from app_analyze.common.kit_config import KitConfig
from app_analyze.scan.sequence.seq_desc import FuncDesc, ObjDesc
from app_analyze.scan.sequence.seq_utils import save_api_seq, sort_apis, is_unused_api, rename_func_name
from app_analyze.scan.sequence.api_filter import GLOBAL_FILTER_PREFIX
from app_analyze.scan.clang_utils import call_expr, skip_implicit, get_attr, get_children
from app_analyze.scan.clang_parser import cuda_enabled, usr_namespace, find_right_angle


# three kinds: 1.invalid, 2.usr_defined, 3.acc_lib
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


# input arguments
def _get_input_args(node):
    args = list()

    refs = node.referenced
    if not refs:
        return args

    parameters = [f'{x.type.spelling}' for x in node.referenced.get_arguments()]
    arguments = list(node.get_arguments())

    for param, x in zip(parameters, arguments):
        # x = skip_implicit(x)
        # if not x:  # 有默认值的Keyword参数，如果实参未传，则为None
        #     continue
        args.append(param)
    return args


# class or struct info
def _get_obj_info(node, func_attr):
    def _get_namespace(c):
        namespace = ''
        if c.parent and get_attr(c.parent, 'referenced.kind') == CursorKind.CLASS_DECL:
            namespace = c.parent.spelling

        if not namespace:
            children = get_children(c)
            for child in children:
                if child.kind == CursorKind.TYPE_REF and get_attr(child, 'referenced.kind') == CursorKind.CLASS_DECL:
                    namespace = child.referenced.spelling
                    break
        return namespace

    info = call_expr(node)
    obj = ObjDesc()
    if info.api.endswith('.' + info.spelling):
        obj.record_name = info.api.replace('.' + info.spelling, '')
    elif info.api.endswith('->' + info.spelling):
        obj.record_name = info.api.replace('->' + info.spelling, '')
    elif info.api.endswith('::' + info.spelling):
        obj.record_name = info.api.replace('::' + info.spelling, '')
    elif info.api == info.spelling:
        obj.record_name = _get_namespace(node)
    else:
        # print('---------------> ' + info.api + ', -------> ' + info.spelling)
        obj = None
    func_attr.obj_info = obj
    if func_attr.return_type == '':
        func_attr.return_type = info.result_type


def _visit_function_decl(node, api_type, arg_dict=None):
    def _format_api(ns, api):
        if ns and api:
            # 拆分模板类型，保留第一个类型，cv::Ptr<cv::cudacodec::VideoReader>
            left_bracket, right_bracket = api.find('<'), find_right_angle('>')
            if left_bracket != -1 and right_bracket != -1:
                api = api[:left_bracket] + api[right_bracket + 1:]

            ns_end = api.rfind('::')
            if ns_end == -1:  # api无命名空间
                api = f'{ns}::{api}'
            else:
                api_ns = api[:ns_end]
                ns_idx = ns.find(api_ns)
                if api_ns.startswith(ns):
                    pass
                elif ns_idx == -1:  # api_ns不在ns里，当然也not ns.startswith(api_ns)，例如cv和Scalar::all
                    api = f'{ns}::{api}'
                elif not ns.startswith(api_ns):  # api.startswith('')为True，例如cv::dnn和dnn::Net
                    api = f'{ns[:ns_idx]}{api}'
        return api

    func_attr = FuncDesc()
    func_attr.func_name = node.spelling
    func_attr.return_type = node.result_type.spelling
    # func_attr.location = node.location
    func_attr.location = location_init(node)
    func_attr.hash_code = node.hash

    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)

    func_attr.is_usr_def = True if api_type == 'usr_defined' else False
    if api_type == 'acc_lib':
        func_attr.acc_name = get_attr(node, 'lib')
        func_attr.func_name = _format_api(arg_dict['usr_ns'], node.spelling)
        rename_func_name(func_attr)

    func_attr.root_file = node.referenced.location.file.name
    if is_unused_api(func_attr):
        func_attr = None
    else:
        func_attr.set_func_id()

    return func_attr


def _visit_cxx_method(node, api_type='invalid'):
    func_attr = FuncDesc()
    func_attr.func_name = node.spelling
    func_attr.return_type = node.result_type.spelling
    # func_attr.location = node.location
    func_attr.location = location_init(node)
    func_attr.hash_code = node.hash

    func_attr.parm_decl_names = _get_input_args(node)
    func_attr.parm_num = len(func_attr.parm_decl_names)

    _get_obj_info(node, func_attr)
    func_attr.is_cxx_method = True
    func_attr.is_usr_def = True if api_type == 'usr_defined' else False
    if api_type == 'acc_lib':
        func_attr.acc_name = get_attr(node, 'lib')
        rename_func_name(func_attr)

    func_attr.root_file = node.referenced.location.file.name
    if is_unused_api(func_attr):
        func_attr = None
    else:
        func_attr.set_func_id()

    return func_attr


def _visit_call_expr(node, rst, pth):
    for c in get_children(node):
        if c.spelling.startswith(GLOBAL_FILTER_PREFIX):
            continue

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
                        func_attr = _visit_function_decl(c, api_type, arg_dict)

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
    if node.spelling.startswith(GLOBAL_FILTER_PREFIX):
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
                    func_attr = _visit_function_decl(node, api_type, arg_dict)

                if not func_attr:
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


def location_init(node):
    location = dict()
    location['column'] = node.location.column
    location['file'] = node.location.file.name
    location['line'] = node.location.line
    location['offset'] = node.location.offset
    return location