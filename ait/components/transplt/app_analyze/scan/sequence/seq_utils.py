class Func:
    def __init__(self):
        self.func_id = 0
        self.func_name = ''
        self.class_name = ''
        self.namespace = ''
        self.para_type = ''


class FuncAttr:
    def __init__(self):
        self.obj_info = None
        self.location = None
        self.func_name = ''
        self.namespace = ''
        self.parm_num = 0
        self.parm_decl_names = ''
        self.return_type = ''
        self.acc_name = ''
        self.is_usr_def = False

        self.hash_code = 0
        self.is_cxx_method = False
        self.has_called = False


class ObjInfo:
    def __init__(self):
        self.cxx_record_name = ''
        self.bases_num = 0
        self.is_polymorphic = False


class SeqDesc:
    def __init__(self):
        self.seq_id = 0
        self.seq = ''
        self.seq_count = 0
        self.seq_location = ''
        self.seq_begin = 0
        self.seq_end = 0
