from app_analyze.common.kit_config import KitConfig

_GLOBAl_EXPERT_LIBS_DICT = dict()


def from_int(x):
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_float(x):
    assert isinstance(x, float) and not isinstance(x, bool)
    return x


def from_str(x):
    assert isinstance(x, str)
    return x


def from_list(f, x):
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c, x):
    assert isinstance(x, c)
    return x.to_dict()


def to_enum(c, x):
    assert isinstance(x, c)
    return x.value


def from_none(x):
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass

    assert False


class Seq:
    def __init__(self, src_seqs, dst_seq, function):
        self.src_seqs = src_seqs
        self.dst_seq = dst_seq
        self.function = function

    @staticmethod
    def from_dict(obj):
        assert isinstance(obj, dict)
        src_seqs = from_list(lambda x: from_list(from_int, x), obj.get('src_seqs'))
        dst_seq = from_list(from_int, obj.get('dst_seq'))
        function = from_str(obj.get('function'))
        return Seq(src_seqs, dst_seq, function)

    def to_dict(self):
        result = dict()
        result['src_seqs'] = from_list(lambda x: from_list(from_int, x), self.src_seqs)
        result['dst_seq'] = from_list(from_int, self.dst_seq)
        result['function'] = from_str(self.function)
        return result


class SeqInfo:
    def __init__(self, seqs):
        self.seqs = seqs

    @staticmethod
    def from_dict(obj):
        assert isinstance(obj, dict)
        seqs = from_list(Seq.from_dict, obj.get('seqs'))
        return SeqInfo(seqs)

    def to_dict(self):
        result = dict()
        result['seqs'] = from_list(lambda x: to_class(Seq, x), self.seqs)
        return result


class ExpertLibs:
    def __init__(self, acc_lib_dict):
        self.acc_lib_dict = acc_lib_dict

    @staticmethod
    def from_dict(obj):
        assert isinstance(obj, dict)
        acc_lib_dict = dict()
        for lib, _ in KitConfig.ACC_LIB_ID_PREFIX.items():
            if obj.get(lib, None):
                lib_content = SeqInfo.from_dict(obj.get(lib))
                acc_lib_dict[lib] = lib_content

        return ExpertLibs(acc_lib_dict)

    def to_dict(self):
        result = dict()
        for lib, _ in KitConfig.ACC_LIB_ID_PREFIX.items():
            result[lib] = to_class(SeqInfo, self.acc_lib_dict[lib])
        return result


def expert_libs_from_dict(s):
    return ExpertLibs.from_dict(s)


def expert_libs_to_dict(x):
    return to_class(ExpertLibs, x)


def get_expert_libs():
    return expert_libs_from_dict(_GLOBAl_EXPERT_LIBS_DICT)


def set_expert_libs(expert_libs):
    _GLOBAl_EXPERT_LIBS_DICT.update(expert_libs)
