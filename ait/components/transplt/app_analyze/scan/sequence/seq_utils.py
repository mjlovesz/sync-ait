from app_analyze.scan.sequence.api_filter import ACC_FILTER


def is_unused_api(func_desc):
    val = ACC_FILTER.get(func_desc.acc_name, None)
    if val:
        acc_file = func_desc.root_file
        file_filter = val.get('file_filter', [])
        if any(acc_file.endswith(p) for p in file_filter):
            return True

        api_name = func_desc.api_name
        api_filter = val.get('api_filter', [])
        if api_name in api_filter:
            return True

    return False


def save_api_seq(seq_desc, result):
    api_cnt = len(seq_desc.api_seq)
    if api_cnt == 1:
        seq_desc.clear()
    elif api_cnt > 1:
        new_seq_desc = seq_desc.trans_to()
        result.append(new_seq_desc)


def sort_apis(seq):
    idx = 0
    cnt = len(seq)
    i = 0
    visited_apis = []
    while i < cnt:
        idx_flag = True
        chk_flag = False

        api = seq[idx]
        if api in visited_apis:
            idx += 1
            continue
        visited_apis.append(api)

        item_idx = len(api[1])
        for j in range(idx + 1, cnt):
            cur_idx = len(seq[j][1])
            if item_idx < cur_idx:
                if j + 1 == cnt:
                    seq.pop(idx)
                    seq.insert(j, api)
                    idx_flag = False
                    break
                else:
                    chk_flag = True
                    continue
            else:
                if chk_flag:
                    seq.pop(idx)
                    seq.insert(j - 1, api)
                    idx_flag = False
                    break
                else:
                    if item_idx == cur_idx:
                        break
        if idx_flag:
            idx += 1
        i += 1
