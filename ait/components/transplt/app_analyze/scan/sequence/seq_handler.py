import collections


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

            seq_desc.debug_string()

    @staticmethod
    def format_api_seqs(seqs):
        def _len_two_lists(arr1, arr2):
            if not arr1 or not arr2:
                return 0
            idx = 0
            while idx < len(arr1) and idx < len(arr2) and arr1[idx].full_name == arr2[idx].full_name:
                idx += 1
            return idx

        result = []
        for seq_desc in seqs:
            sub_rst = []
            seq_len = len(seq_desc.api_seq)
            max_len = 0
            if seq_len <= 1:
                continue

            arrs = []  # 存放S的后缀字符串
            for i in range(0, seq_len):
                arrs.append(seq_desc.api_seq[seq_len - 1 - i:])

            # 两个相邻字符串的最长公共前缀
            for i in range(0, seq_len - 1):
                for j in range(i + 1, seq_len):
                    tmp = _len_two_lists(arrs[i], arrs[j])
                    if tmp > max_len:
                        max_len = tmp
                        sub_rst = arrs[i][0:max_len]

            if sub_rst:
                result.append(sub_rst)
        return result
