import pickle

from app_analyze.utils.log_util import logger
from app_analyze.scan.sequence.aprioriv2 import apriori
from app_analyze.scan.sequence.seq_desc import GLOBAl_FUNC_ID_DICT
from app_analyze.scan.sequence.acc_libs import GLOBAl_EXPERT_LIBS_DICT
from app_analyze.scan.sequence.prefix_span import prefixspan


class SeqHandler:
    @staticmethod
    def union_api_seqs(seqs):
        def _get_union_api(seq_obj, api_seq):
            for api in seq_obj.api_seq:
                if not api.is_usr_def:
                    api_seq.append(api)
                else:
                    # if usr defined api in seq, check if api can union
                    usr_api = usr_def_dict.get(api.unique_name, None)
                    if not usr_api:
                        api_seq.append(api)
                        continue

                    usr_api['key'].has_called = True
                    _get_union_api(usr_api['key'], api_seq)

        if len(seqs) == 1:
            if all(not p.is_usr_def for p in seqs[0].api_seq):
                seqs[0].has_usr_def = False
            return

        usr_def_dict = dict()
        for seq_desc in seqs:
            usr_def_dict[seq_desc.entry_api.unique_name] = {'key': seq_desc}

        for seq_desc in seqs:
            if not seq_desc.has_usr_def:
                continue

            new_api_seq = []
            _get_union_api(seq_desc, new_api_seq)
            seq_desc.api_seq = new_api_seq
            if all(not p.is_usr_def for p in new_api_seq):
                seq_desc.has_usr_def = False

    @staticmethod
    def clean_api_seqs(seqs, deep_flag):
        def _compact_apis(api_seq):
            apis = []
            pre_api_name = None
            for api in api_seq:
                if not pre_api_name:
                    pre_api_name = api.unique_name
                    apis.append(api)
                else:
                    if pre_api_name == api.unique_name:
                        continue

                    apis.append(api)
                    pre_api_name = api.unique_name
            return apis

        rst = []
        for seq_desc in seqs:
            if seq_desc.has_called:
                continue

            if deep_flag:
                if not seq_desc.has_usr_def:
                    # all acc lib apis in seq
                    seq_desc.api_seq = _compact_apis(seq_desc.api_seq)
                    rst.append(seq_desc)

                    logger.info(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
                    seq_desc.debug_string()
                    continue

                new_api_seq = []
                # delete use define api
                while seq_desc.api_seq:
                    func_desc = seq_desc.api_seq.pop(0)
                    if not func_desc.is_usr_def:
                        new_api_seq.append(func_desc)
                # deduplicate apis, eg: a b b b c --> a b c
                seq_desc.api_seq = _compact_apis(new_api_seq)
                seq_desc.has_usr_def = False

            rst.append(seq_desc)
            logger.info(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
            seq_desc.debug_string()

        return rst

    @staticmethod
    def _store_api_seqs(seqs, path='./'):
        seqs_txt = path + 'seqs.tmp.bin'
        with open(seqs_txt, 'wb') as f:
            pickle.dump(seqs, f)

        seqs_idx_txt = path + 'seqs_idx.tmp.bin'
        id_dict = dict(zip(GLOBAl_FUNC_ID_DICT.values(), GLOBAl_FUNC_ID_DICT.keys()))
        with open(seqs_idx_txt, 'wb') as f:
            pickle.dump(id_dict, f)

    @staticmethod
    def load_api_seqs(path):
        with open(path, 'rb') as text:
            data = pickle.load(text)
        return data

    @staticmethod
    def debug_string(seqs, idx_dict=None):
        if GLOBAl_FUNC_ID_DICT:
            idx_dict = dict(zip(GLOBAl_FUNC_ID_DICT.values(), GLOBAl_FUNC_ID_DICT.keys()))

        assert idx_dict
        rst_str = 'The sequences result are: \n'
        for i, seq in enumerate(seqs):
            d_str = []
            for idx in seq:
                d_str.append(idx_dict[idx])
            rst_str += str(i) + '. ' + '-->'.join(d_str) + '\n'

        logger.info(f'{rst_str}')

    @staticmethod
    def _dedup_api_seqs(seqs):
        import re
        for seq_desc in seqs:
            cnt = len(seq_desc.api_seq)
            whole = [str(_.func_id) for _ in seq_desc.api_seq]
            whole_str = ''.join(whole)

            rst = {}
            for i in range(cnt):
                for j in range(cnt - i, -1, -1):
                    sub_str = ')('.join(whole[i:j])
                    sub_str = '(' + sub_str + ')'
                    matchers = re.finditer(sub_str, whole_str)
                    vals = list()
                    for m in matchers:
                        vals.append((m.start(), m.end()))

                    if len(vals) == 1:
                        continue
                    else:
                        cnt = 0
                        for k in range(len(vals) - 1):
                            if vals[k][1] == vals[k + 1][0]:
                                cnt += 1

                        rst[(i, j)] = cnt

    @staticmethod
    def mining_one_seq(seqs):
        def _len_two_lists(arr1, arr2):
            if not arr1 or not arr2:
                return 0
            idx = 0
            while idx < len(arr1) and idx < len(arr2) and arr1[idx] == arr2[idx]:
                idx += 1
            return idx

        result = []
        for seq in seqs:
            seq_len = len(seq)

            max_len = 1
            if seq_len <= 1:
                continue

            arrays = []  # 存放S的后缀字符串
            for i in range(0, seq_len):
                arrays.append(seq[seq_len - 1 - i:])

            # 两个相邻字符串的最长公共前缀
            for i in range(0, seq_len - 1):
                for j in range(i + 1, seq_len):
                    tmp = _len_two_lists(arrays[i], arrays[j])
                    if tmp <= max_len:
                        continue

                    # max_len = tmp
                    sub_rst = arrays[i][0:tmp]

                    if sub_rst and len(sub_rst) >= 4:
                        result.append(sub_rst)
        return result

    @staticmethod
    def mining_api_seqs(seqs):
        result = []

        # min_support = 0.2
        # l1, support_data1 = apriori(seqs, min_support)
        # print(f'L({min_support}): ', l1)
        # for vals in l1:
        #     for seq in vals:
        #         if len(seq) > 1:
        #             result.append(seq)

        top_k = 200
        l1, freq = prefixspan(seqs, top_k)
        for item in l1:
            seq = item[1]
            if len(seq) >= 4:
                result.append(seq)
        return result

    def format_api_seqs(self, seqs):
        rst = []
        for seq_desc in seqs:
            cur_idx_list = [_.func_id for _ in seq_desc.api_seq]
            if cur_idx_list:
                rst.append(cur_idx_list)

        self._store_api_seqs(rst)
        return rst


def get_idx_tbl():
    idx_dict = dict(zip(GLOBAl_FUNC_ID_DICT.values(), GLOBAl_FUNC_ID_DICT.keys()))
    return idx_dict


def set_idx_tbl(idx_dict):
    GLOBAl_FUNC_ID_DICT.update(dict(zip(idx_dict.values(), idx_dict.keys())))


def set_expert_libs_tbl(expert_libs):
    GLOBAl_EXPERT_LIBS_DICT.update(expert_libs)


def mining_api_seqs(seqs, idx_seq_dict=None):
    all_seqs = set()

    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.info('===============Sequences Before Mining===============')
    handler.debug_string(seqs, idx_seq_dict)

    dup_apis = handler.mining_one_seq(seqs)
    for apis in dup_apis:
        if len(set(apis)) == len(apis):
            all_seqs.add(tuple(apis))

    dig_apis = handler.mining_api_seqs(seqs)
    for apis in dig_apis:
        all_seqs.add(tuple(apis))

    # for apis in seqs:
    #     if len(apis) > 3:
    #         all_seqs.add(tuple(apis))

    logger.info('===============Sequences After Mining===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


def load_api_seqs(seq_file):
    seq_info = SeqHandler.load_api_seqs(seq_file)
    return seq_info


if __name__ == "__main__":
    api_seqs = load_api_seqs('../../model/seqs.bin')
    idx_seqs = load_api_seqs('../../model/seqs_idx.bin')
    mining_api_seqs(api_seqs, idx_seqs)
