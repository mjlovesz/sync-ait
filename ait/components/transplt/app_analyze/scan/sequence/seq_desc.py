from app_analyze.utils.log_util import logger


class FuncDesc:
    def __init__(self):
        self.obj_info = None
        self.location = None
        self.acc_name = ''
        self.namespace = ''
        self.func_name = ''
        self.parm_num = 0
        self.parm_decl_names = []
        self.return_type = ''
        self.is_usr_def = False

        self.hash_code = 0
        self.is_cxx_method = False

    def get_full_name(self):
        api_str = self.get_api_name()
        arg_str = self.get_arg_name()
        if arg_str:
            api_str += '(' + arg_str + ')'
        return api_str

    def get_api_name(self):
        if self.obj_info:
            vals = [self.obj_info.record_name, self.func_name]
        else:
            if self.namespace:
                vals = [self.namespace, self.func_name]
            else:
                vals = [self.func_name]

        return '.'.join(vals)

    def get_arg_name(self):
        return ','.join(self.parm_decl_names)


class ObjDesc:
    def __init__(self):
        self.record_name = ''
        self.bases_num = 0
        self.is_polymorphic = False


class SeqDesc:
    def __init__(self):
        self.seq_id = 0
        self.seq = ''
        self.seq_count = 0
        self.entry_api = ''
        self.api_seq = []
        self.has_usr_def = False
        self.has_called = False

    def trans_to(self):
        new_desc = SeqDesc()
        new_desc.entry_api = self.api_seq[0]
        new_desc.api_seq.extend(self.api_seq[1:])
        new_desc.has_usr_def = self.has_usr_def
        self.clear()
        return new_desc

    def clear(self):
        self.api_seq = []
        self.has_usr_def = False

    def debug_string(self):
        rst = 'Entry Function is: ' + self.entry_api.get_api_name() + '\n'
        apis = [_.get_full_name() for _ in self.api_seq]
        rst += '-->'.join(apis)
        logger.info(rst)
