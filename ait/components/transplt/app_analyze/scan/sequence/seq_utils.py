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

from app_analyze.scan.sequence.api_filter import ACC_FILTER


def is_unused_api(func_desc):
    val = ACC_FILTER.get(func_desc.acc_name, None)
    if val is None:
        return False

    acc_file = func_desc.root_file
    file_filter = val.get('file_filter', [])
    if any(acc_file.endswith(p) for p in file_filter):
        return True

    api_name = func_desc.api_name
    api_filter = val.get('api_filter', [])
    if api_name in api_filter:
        return True

    return False


def rename_func_name(func_desc):
    val = ACC_FILTER.get(func_desc.acc_name, None)
    if val is None:
        return

    ns_filter = val.get('namespace_filter', {})
    func_name = func_desc.func_name
    record_name = func_desc.obj_info.record_name if func_desc.obj_info is not None else ''
    for ns_prefix, ns in ns_filter.items():
        if func_name.startswith(ns_prefix):
            name = func_name
            flag = True
        elif record_name.startswith(ns_prefix):
            name = record_name
            flag = False
        else:
            continue

        left = name.replace(ns_prefix, '')
        if left.startswith('::'):
            if flag:
                func_desc.func_name = ns + left
            else:
                func_desc.obj_info.record_name = ns + left
        else:
            pos = left.find('::')
            if flag:
                func_desc.func_name = ns + left[pos:]
            else:
                func_desc.obj_info.record_name = ns + left[pos:]


def save_api_seq(seq_desc, result):
    api_cnt = len(seq_desc.api_seq)
    if api_cnt == 1:
        seq_desc.clear()
    elif api_cnt > 1:
        new_seq_desc = seq_desc.trans_to()
        result.append(new_seq_desc)


def sort_apis(seq):
    idx = 0
    visited_apis = list()
    cnt = len(seq)
    i = 0
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
                elif item_idx == cur_idx:
                    break

        if idx_flag:
            idx += 1
        i += 1
