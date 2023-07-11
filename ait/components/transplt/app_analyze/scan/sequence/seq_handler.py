import pickle

from app_analyze.utils.log_util import logger
from app_analyze.scan.sequence.aprioriv2 import apriori


class SeqHandler:
    @staticmethod
    def union_api_seqs(seqs):
        def _get_union_api(seq_obj, api_seq):
            for api in seq_obj.api_seq:
                if not api.is_usr_def:
                    api_seq.append(api)
                else:
                    # if usr defined api in seq, check if api can union
                    usr_api = usr_def_dict.get(api.full_name, None)
                    if not usr_api:
                        api_seq.append(api)
                        continue

                    usr_api['key'].has_called = True
                    _get_union_api(usr_api['key'], api_seq)

        usr_def_dict = dict()
        for seq_desc in seqs:
            usr_def_dict[seq_desc.entry_api.full_name] = {'key': seq_desc}

        for seq_desc in seqs:
            if not seq_desc.has_usr_def:
                continue

            new_api_seq = []
            _get_union_api(seq_desc, new_api_seq)
            seq_desc.api_seq = new_api_seq
            if all(not p.is_usr_def for p in new_api_seq):
                seq_desc.has_usr_def = False

    @staticmethod
    def clean_api_seqs(seqs):
        def _dedup_apis(api_seqs):
            apis = []
            pre_api_name = None
            for api in api_seqs:
                if not pre_api_name:
                    pre_api_name = api.full_name
                    apis.append(api)
                else:
                    if pre_api_name == api.full_name:
                        continue

                    apis.append(api)
                    pre_api_name = api.full_name
            return apis

        rst = []
        for seq_desc in seqs:
            if seq_desc.has_called:
                continue

            if not seq_desc.has_usr_def:
                seq_desc.api_seq = _dedup_apis(seq_desc.api_seq)
                rst.append(seq_desc)

                logger.info(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
                seq_desc.debug_string()
                continue

            new_api_seq = []
            while seq_desc.api_seq:
                func_desc = seq_desc.api_seq.pop(0)
                if not func_desc.is_usr_def:
                    new_api_seq.append(func_desc)
            seq_desc.api_seq = _dedup_apis(new_api_seq)
            seq_desc.has_usr_def = False
            rst.append(seq_desc)

            logger.info(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
            seq_desc.debug_string()

        return rst

    @staticmethod
    def cluster_api_seqs(seqs):
        rst = []
        for seq_desc in seqs:
            rst.append([_.func_id for _ in seq_desc.api_seq])

        # text_txt = 'api_seqs.txt'
        # with open(text_txt, 'wb') as f:
        #     pickle.dump(rst, f)

        l1, support_data1 = apriori(rst, min_support=0.7)
        print('L(0.7): ', l1)
        print('supportData(0.7): ', support_data1)
        return l1

    @staticmethod
    def dedup_api_seqs(seqs):
        import re
        for seq_desc in seqs:
            cnt = len(seq_desc.api_seq)
            whole = [str(_.func_id) for _ in seq_desc.api_seq]
            whole_str = '_'.join(whole)
            for i in range(cnt):
                for j in range(cnt - i, -1, -1):
                    sub_str = '_'.join(whole[i:j])
                    matchers = re.finditer(sub_str, whole_str)
                    vals = list()
                    for m in matchers:
                        vals.append((m.start(), m.end()))

                    if len(vals) == 1:
                        continue
                    # print()

    @staticmethod
    def format_api_seqs(seqs):
        def _len_two_lists(arr1, arr2):
            if not arr1 or not arr2:
                return 0
            idx = 0
            while idx < len(arr1) and idx < len(arr2) and arr1[idx].func_id == arr2[idx].func_id:
                idx += 1
            return idx

        result = []
        for seq_desc in seqs:
            sub_rst = []
            seq_len = len(seq_desc.api_seq)
            max_len = 0
            if seq_len <= 1:
                continue

            arrays = []  # 存放S的后缀字符串
            for i in range(0, seq_len):
                arrays.append(seq_desc.api_seq[seq_len - 1 - i:])

            # 两个相邻字符串的最长公共前缀
            for i in range(0, seq_len - 1):
                for j in range(i + 1, seq_len):
                    tmp = _len_two_lists(arrays[i], arrays[j])
                    if tmp > max_len:
                        max_len = tmp
                        sub_rst = arrays[i][0:max_len]

            if sub_rst:
                result.append(sub_rst)
        return result
