import pickle
import numpy as np

from app_analyze.utils.log_util import logger
from app_analyze.common.kit_config import SeqArgs
from app_analyze.scan.sequence.aprioriv2 import apriori
from app_analyze.scan.sequence.seq_desc import get_idx_tbl
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
                    usr_api = usr_def_dict.get(api.full_name, None)
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
    def clean_api_seqs(seqs, deep_flag):
        def _compact_apis(api_seq):
            apis = []
            pre_api_id = None
            for api in api_seq:
                if not pre_api_id:
                    pre_api_id = api.func_id
                    apis.append(api)
                else:
                    if pre_api_id == api.func_id:
                        continue

                    apis.append(api)
                    pre_api_id = api.func_id
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
    def store_api_seqs(seqs, id_dict=None, path='./'):
        seqs_txt = path + 'seqs.tmp.bin'
        with open(seqs_txt, 'wb') as f:
            pickle.dump(seqs, f)

        seqs_idx_txt = path + 'seqs_idx.tmp.bin'
        if not id_dict:
            id_dict = get_idx_tbl()

        with open(seqs_idx_txt, 'wb') as f:
            pickle.dump(id_dict, f)

    @staticmethod
    def load_api_seqs(path):
        with open(path, 'rb') as text:
            data = pickle.load(text)
        return data

    @staticmethod
    def debug_string(seqs, idx_dict=None):
        if not idx_dict:
            idx_dict = get_idx_tbl()

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

                    sub_rst = arrays[i][0:tmp]

                    if sub_rst and len(sub_rst) >= SeqArgs.SEQ_MIN_LEN:
                        result.append(sub_rst)
        return result

    @staticmethod
    def mining_api_seqs(seqs):
        result = []

        # l1, support_data1 = apriori(seqs, SeqArgs.APRIORI_MIN_SUPPORT)
        # print(f'L({SeqArgs.APRIORI_MIN_SUPPORT}): ', l1)
        # for vals in l1:
        #     for seq in vals:
        #         if len(seq) >= SeqArgs.SEQ_MIN_LEN:
        #             result.append(seq)

        l1, freq = prefixspan(seqs, SeqArgs.PREFIX_SPAN_TOP_K)
        for item in l1:
            seq = item[1]
            if len(seq) >= SeqArgs.SEQ_MIN_LEN:
                result.append(seq)
        return result

    def format_api_seqs(self, seqs):
        rst = []
        for seq_desc in seqs:
            cur_idx_list = [_.func_id for _ in seq_desc.api_seq]
            if cur_idx_list:
                rst.append(cur_idx_list)

        self.store_api_seqs(rst)
        return rst

    @staticmethod
    def group_api_seqs(seqs, expert_libs):
        # a:描出的序列; b:模板序列
        def _calc_dist(a, b):
            intersection = np.intersect1d(a, b)
            union = np.union1d(a, b)
            ratio = len(intersection) * 1.0 / len(union) if len(intersection) else -1
            # in_flag为true，表示模板序列包含在扫描出来的序列中
            in_flag = True if len(intersection) == len(b) else False
            return ratio, in_flag

        def _group(seq, lib_seqs, sim):
            # 遍历对应加速库的模板
            for lib_seq in lib_seqs:
                # 获取加速库的模板序列
                src_seq = lib_seq.src_seq
                # 计算扫描出的序列跟模板序列的匹配度
                val, flag = _calc_dist(seq, src_seq)
                if flag:
                    # 模板序列包含在扫描出来的序列中,可以进行推荐
                    in_lib_seqs[lib_seq] = (val, src_seq)
                else:
                    # 小于阈值的，不进行推荐
                    if val < SeqArgs.SIM_MIN_SUPPORT:
                        continue

                    if val == sim:
                        # 具有相同阈值的序列都保存起来
                        cs_lib_seqs[lib_seq] = (sim, src_seq)
                    elif val > sim:
                        # 如果当前相似度大于阈值，进行最新序列的替换
                        sim = val
                        cs_lib_seqs.clear()
                        cs_lib_seqs[lib_seq] = (sim, src_seq)
            return sim

        def _sort(usr_seq, cs_seqs, in_seqs):
            if cs_seqs and not in_seqs:
                # 序列和模板库交叉
                return dict(zip(cs_seqs.keys(), [_[0] for _ in list(cs_seqs.values())]))
            elif not cs_seqs and not in_seqs:
                # 没有推荐的序列
                return dict()
            elif not cs_seqs and in_seqs:
                # 序列包含在模板库中
                return dict(zip(in_seqs.keys(), [_[0] for _ in list(in_seqs.values())]))
            else:
                # 既存在序列在模板库中的情况，又存在序列跟模板库交叉的情况
                apis = list()
                for _ in list(in_seqs.values()):
                    apis += _[1]

                # 如果扫描到的序列包括多个子序列，并且多个子序列拼起来的相似度高于交叉序列的相似度，
                # 则选择包含的推荐方式，否则选择交叉的推荐方式。
                merged_ratio, _ = _calc_dist(usr_seq, list(set(apis)))
                max_cs_ratio = list(cs_seqs.values())[0][0]

                if merged_ratio > max_cs_ratio:
                    return dict(zip(in_seqs.keys(), [_[0] for _ in list(in_seqs.values())]))
                else:
                    return dict(zip(cs_seqs.keys(), [_[0] for _ in list(cs_seqs.values())]))

        result = dict()
        for seq_desc in seqs:
            sim_val = -1
            # 模板序列和扫描出来的序列是交叉关系
            cs_lib_seqs = dict()
            # 模板序列包含在扫描出来的序列中
            in_lib_seqs = dict()

            # 获取序列中所涉及的加速库
            acc_names = set([_.acc_name for _ in seq_desc.api_seq if _.acc_name])
            # 获取当前序列的id列表
            cur_idx_list = [_.func_id for _ in seq_desc.api_seq]
            for acc_name in acc_names:
                # 获取当前加速库对应的模板
                seq_info = expert_libs.acc_lib_dict.get(acc_name, None)
                if seq_info:
                    # 进行模板匹配
                    sim_val = _group(cur_idx_list, seq_info.seqs, sim_val)
            # 对扫描出来的结果进行处理
            result[seq_desc] = _sort(cur_idx_list, cs_lib_seqs, in_lib_seqs)
        return result


def filter_api_seqs(seqs, idx_seq_dict=None):
    all_seqs = list()
    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.info('===============Sequences Before Filtering===============')
    handler.debug_string(seqs, idx_seq_dict)

    for seq in seqs:
        if len(seq) >= SeqArgs.SEQ_MIN_LEN:
            all_seqs.append(seq)

    logger.info('===============Sequences After Filtering===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


def mining_api_seqs(seqs, idx_seq_dict=None):
    handler = SeqHandler()

    # new_idx_seq_dict = dict()
    # base = KitConfig.ACC_LIB_ID_PREFIX['mxBase'] * KitConfig.ACC_ID_BASE
    # for idx, name in idx_seq_dict.items():
    #     new_idx_seq_dict[idx + base] = name
    # new_api_seqs = []
    # for seq in seqs:
    #     new_api_seqs.append([base + _ for _ in seq])
    # handler.store_api_seqs(new_api_seqs, new_idx_seq_dict)

    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.info('===============Sequences Before Mining===============')
    handler.debug_string(seqs, idx_seq_dict)

    all_seqs = set()
    dup_apis = handler.mining_one_seq(seqs)
    for apis in dup_apis:
        if len(set(apis)) == len(apis):
            all_seqs.add(tuple(apis))

    dig_apis = handler.mining_api_seqs(seqs)
    for apis in dig_apis:
        all_seqs.add(tuple(apis))

    logger.info('===============Sequences After Mining===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


def load_api_seqs(seq_file: object) -> object:
    seq_info = SeqHandler.load_api_seqs(seq_file)
    return seq_info


if __name__ == "__main__":
    api_seqs = load_api_seqs('../../model/seqs.bin')
    idx_seqs = load_api_seqs('../../model/seqs_idx.bin')
    mining_api_seqs(api_seqs, idx_seqs)
