from app_analyze.utils.log_util import logger

GLOBAl_FUNC_ID_DICT = dict()
FUNC_ID_COUNTER = -1

GLOBAL_FILE_ID_DICT = dict()
FILE_ID_COUNTER = -1


class FuncDesc:
    def __init__(self):
        self.obj_info = None
        self.location = None
        self.acc_name = ''
        self.namespace = ''
        self.func_name = ''
        self.parm_num = 0
        self.parm_decl_names = list()
        self.return_type = ''
        self.is_usr_def = False
        self.root_file = None  # which file function define

        self.hash_code = 0
        self.is_cxx_method = False

    @property
    def unique_name(self):
        full_str = self.full_name
        if self.acc_name:
            full_str = '[' + self.acc_name + ']' + full_str
        return full_str

    @property
    def full_name(self):
        api_str = self.api_name
        arg_str = self.arg_name
        if arg_str:
            api_str += '(' + arg_str + ')'
        return api_str

    @property
    def api_name(self):
        if self.obj_info:
            names = [self.obj_info.record_name, self.func_name]
        else:
            if self.namespace:
                names = [self.namespace, self.func_name]
            else:
                names = [self.func_name]

        return '.'.join(names)

    @property
    def arg_name(self):
        return ','.join(self.parm_decl_names)

    @property
    def func_id(self):
        if self.is_usr_def:
            return -1

        fid = GLOBAl_FUNC_ID_DICT.get(self.unique_name, None)
        if fid is None:
            global FUNC_ID_COUNTER
            FUNC_ID_COUNTER += 1
            fid = FUNC_ID_COUNTER
            GLOBAl_FUNC_ID_DICT[self.unique_name] = fid
        return fid

    @property
    def file_id(self):
        if not self.is_usr_def:
            return -1

        fid = GLOBAL_FILE_ID_DICT.get(self.root_file, None)
        if fid is None:
            global FILE_ID_COUNTER
            FILE_ID_COUNTER += 1
            fid = FILE_ID_COUNTER
            GLOBAL_FILE_ID_DICT[self.root_file] = fid
        return fid


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
        self.api_seq = list()
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
        self.api_seq = list()
        self.has_usr_def = False

    @staticmethod
    def seq_str(api_seq):
        apis = [_.unique_name for _ in api_seq]
        rst = '-->'.join(apis)
        return rst

    def debug_string(self):
        rst = 'Entry Function is: ' + self.entry_api.api_name + '\n'
        apis = [_.unique_name for _ in self.api_seq]
        rst += '-->'.join(apis)
        logger.info(rst)
