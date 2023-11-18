# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from app_analyze.utils.log_util import logger
from app_analyze.utils.io_util import IOUtil
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
    def clean_api_seqs(seqs, infer_flag):
        def _compact_apis(api_seq):
            apis = list()
            pre_api_id = None
            for api in api_seq:
                if not pre_api_id:
                    pre_api_id = api.func_id
                    apis.append(api)
                elif pre_api_id == api.func_id:
                    continue
                else:
                    apis.append(api)
                    pre_api_id = api.func_id
            return apis

        if infer_flag:
            rst = [seq_desc for seq_desc in seqs if not seq_desc.has_called]
            return rst

        rst = list()
        for seq_desc in seqs:
            if seq_desc.has_called:
                continue

            if not seq_desc.has_usr_def:
                # all acc lib apis in seq
                seq_desc.api_seq = _compact_apis(seq_desc.api_seq)
                rst.append(seq_desc)

                logger.debug(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
                seq_desc.debug_string()
                continue

            # delete use define api
            new_api_seq = [func_desc for func_desc in seq_desc.api_seq if not func_desc.is_usr_def]
            # deduplicate apis, eg: a b b b c --> a b c
            seq_desc.api_seq = _compact_apis(new_api_seq)
            seq_desc.has_usr_def = False
            rst.append(seq_desc)
            logger.debug(f'After clean seqs, api seqs length is {len(seq_desc.api_seq)}, the api seq is: ')
            seq_desc.debug_string()

        return rst

    @staticmethod
    def _store_api_seqs(seqs, id_dict=None, path='./'):
        seqs_file = path + 'seqs.tmp.bin'
        IOUtil.bin_safe_dump(seqs, seqs_file)

        seqs_idx_file = path + 'seqs_idx.tmp.bin'
        if not id_dict:
            id_dict = get_idx_tbl()
        IOUtil.bin_safe_dump(id_dict, seqs_idx_file)

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

        logger.debug(f'{rst_str}')

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

        self._store_api_seqs(rst)
        return rst


def filter_api_seqs(seqs, idx_seq_dict=None):
    all_seqs = list()
    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.debug('===============Sequences Before Filtering===============')
    handler.debug_string(seqs, idx_seq_dict)

    for seq in seqs:
        if len(seq) >= SeqArgs.SEQ_MIN_LEN:
            all_seqs.append(seq)

    logger.debug('===============Sequences After Filtering===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


def mining_api_seqs(seqs, idx_seq_dict=None):
    handler = SeqHandler()
    if not idx_seq_dict:
        seqs = handler.format_api_seqs(seqs)

    logger.debug('===============Sequences Before Mining===============')
    handler.debug_string(seqs, idx_seq_dict)

    all_seqs = set()
    dup_apis = handler.mining_one_seq(seqs)
    for apis in dup_apis:
        if len(set(apis)) == len(apis):
            all_seqs.add(tuple(apis))

    dig_apis = handler.mining_api_seqs(seqs)
    for apis in dig_apis:
        all_seqs.add(tuple(apis))

    logger.debug('===============Sequences After Mining===============')
    handler.debug_string(all_seqs, idx_seq_dict)
    return all_seqs


if __name__ == "__main__":
    api_seqs = IOUtil.bin_safe_load('../../model/seqs.bin')
    idx_seqs = IOUtil.bin_safe_load('../../model/seqs_idx.bin')
    mining_api_seqs(api_seqs, idx_seqs)
