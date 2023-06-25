import os
import time

from app_analyze.scan.clang_parser import *
from app_analyze.scan.sequence.ast_visitor import visit
from app_analyze.common.kit_config import KitConfig
from app_analyze.utils.log_util import logger


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
            hit = False
            if not getattr(node, 'scanned', False):
                hit, func_attr = visit(node, sub_rst, result)

            cuda_en = False
            hit = hit and not getattr(node, 'implicit', False)
            # if hit:
            #     api = MACRO_MAP.get(func_attr.func_name, func_attr.func_name)
            #     loc = f"{get_attr(node, 'extent.start.file.name')}, {get_attr(node, 'extent.start.line')}:" \
            #           f"{get_attr(node, 'extent.start.column')}"
            #     args = parse_args(node)
            #     item = {
            #         KitConfig.ACC_API: api,
            #         KitConfig.CUDA_EN: cuda_en,
            #         KitConfig.LOCATION: loc,
            #         KitConfig.CONTEXT: args,
            #         KitConfig.ACC_LIB: get_attr(node, 'lib'),
            #     }
            #     RESULTS.append(item)

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
