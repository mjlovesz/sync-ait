from app_analyze.common.kit_config import KitConfig

GLOBAl_EXPERT_LIBS_DICT = dict()


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
    def __init__(self, src_seqs, dst_seq):
        self.src_seqs = src_seqs
        self.dst_seq = dst_seq

    @staticmethod
    def from_dict(obj):
        assert isinstance(obj, dict)
        src_seqs = from_list(lambda x: from_list(from_int, x), obj.get('src_seqs'))
        dst_seq = from_list(from_int, obj.get('dst_seq'))
        return Seq(src_seqs, dst_seq)

    def to_dict(self):
        result = dict()
        result['src_seqs'] = from_list(lambda x: from_list(from_int, x), self.src_seqs)
        result['dst_seq'] = from_list(from_int, self.dst_seq)
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
        self.opencv = acc_lib_dict.get(KitConfig.OPENCV, None)

    @staticmethod
    def from_dict(obj):
        assert isinstance(obj, dict)
        acc_lib_dict = dict()
        if obj.get('opencv', None):
            opencv = SeqInfo.from_dict(obj.get('opencv'))
            acc_lib_dict['opencv'] = opencv

        return SeqInfo(acc_lib_dict)

    def to_dict(self):
        result = dict()
        if self.opencv:
            result['opencv'] = to_class(SeqInfo, self.opencv)
        return result


def expert_libs_from_dict(s):
    return ExpertLibs.from_dict(s)


def expert_libs_to_dict(x):
    return to_class(ExpertLibs, x)
