from app_analyze.utils.log_util import logger
from app_analyze.common.kit_config import KitConfig

_GLOBAl_FUNC_ID_DICT = dict()

_GLOBAL_FILE_ID_DICT = dict()
_FILE_ID_COUNTER = -1


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

        self._func_id = None

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

        return '::'.join(names)

    @property
    def arg_name(self):
        return ','.join(self.parm_decl_names)

    def set_func_id(self):
        if self.is_usr_def:
            self._func_id = -1
        else:
            name_id_tbl = _GLOBAl_FUNC_ID_DICT.get(self.acc_name, None)
            no_fid_flag = False
            if name_id_tbl:
                self._func_id = name_id_tbl.get(self.full_name, None)  # TODO, file_name + full_name
                if self._func_id is None:
                    no_fid_flag = True
            else:
                _GLOBAl_FUNC_ID_DICT[self.acc_name] = {}
                no_fid_flag = True

            if no_fid_flag:
                base = KitConfig.ACC_LIB_ID_PREFIX[self.acc_name] * KitConfig.ACC_ID_BASE
                offset = len(_GLOBAl_FUNC_ID_DICT[self.acc_name])
                self._func_id = base + offset
                _GLOBAl_FUNC_ID_DICT[self.acc_name][self.full_name] = self._func_id

    @property
    def func_id(self):
        return self._func_id

    @property
    def file_id(self):
        if not self.is_usr_def:
            return -1

        fid = _GLOBAL_FILE_ID_DICT.get(self.root_file, None)
        if fid is None:
            global _FILE_ID_COUNTER
            _FILE_ID_COUNTER += 1
            fid = _FILE_ID_COUNTER
            _GLOBAL_FILE_ID_DICT[self.root_file] = fid
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

    def debug_string(self):
        rst = 'Entry Function is: ' + self.entry_api.api_name + '\n'
        apis = [_.full_name for _ in self.api_seq]
        rst += '-->'.join(apis)
        logger.info(rst)


def get_idx_tbl():
    idx_dict = dict()
    for _, val in _GLOBAl_FUNC_ID_DICT.items():
        idx_dict.update(dict(zip(val.values(), val.keys())))

    return idx_dict


def get_api_lut():
    return _GLOBAl_FUNC_ID_DICT


def set_api_lut(idx_dict):
    base_id_dict = dict(zip(KitConfig.ACC_LIB_ID_PREFIX.values(), KitConfig.ACC_LIB_ID_PREFIX.keys()))
    for idx, name in idx_dict.items():
        res = idx // KitConfig.ACC_ID_BASE
        acc_name = base_id_dict[res]

        acc_libs = _GLOBAl_FUNC_ID_DICT.get(acc_name, None)
        if acc_libs:
            _GLOBAl_FUNC_ID_DICT[acc_name][name] = idx
        else:
            _GLOBAl_FUNC_ID_DICT[acc_name] = {name: idx}
