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

from functools import reduce
from typing import Dict, List
from copy import deepcopy


def apriori(tms: list, min_sup: float, level: int = -1) -> Dict[str, List[List[str]]]:
    cur_collection = {}
    n_sup = int(len(tms) * min_sup)

    # get all the unique element in the item set
    unique_items = reduce(lambda x, y: x | y, (set(items) for items in tms))
    # initialise the C
    unique_tokens = [[item] for item in unique_items]
    unique_tokens.sort(key=lambda x: x[0])

    item_sets = [set(items) for items in tms]

    for iter_num, _ in enumerate(unique_items):
        # step1: count and filter the frequent item sets
        cnt_fr_list = list()
        for i, item in enumerate(unique_tokens):
            item = set(item)
            count = 0
            for item_set in item_sets:
                if item & item_set == item:  # this way, item is subset to item set
                    count += 1

            if count >= n_sup:
                item = list(item)
                item.sort()  # sort is necessary
                cnt_fr_list.append(item)

        if len(cnt_fr_list):
            cur_collection["L" + str(iter_num + 1)] = cnt_fr_list
            if level > 0 and (iter_num + 1 == level):
                return cur_collection

        # step2: check and connect between frequent item sets
        fr_kl = len(cnt_fr_list)
        no_trim_c = set()
        for i in range(fr_kl - 1):
            for j in range(i + 1, fr_kl):
                if cnt_fr_list[i][:-1] != cnt_fr_list[j][:-1]:
                    continue

                temp = deepcopy(cnt_fr_list[i][:-1])
                a, b = cnt_fr_list[i][-1], cnt_fr_list[j][-1]
                temp += [a, b] if a < b else [b, a]
                no_trim_c.add(tuple(temp))  # list is not hashable

        if len(no_trim_c) == 0:
            break

        # step3: truncate the L
        if iter_num == 0:
            unique_tokens = [list(item_set) for item_set in no_trim_c]
            continue

        unique_tokens = []
        cnt_fr_list = {tuple(item_set) for item_set in cnt_fr_list}
        for item_set in no_trim_c:
            item_set = sorted(list(item_set))
            for item in item_set:
                temp_item_set = deepcopy(item_set)
                temp_item_set.remove(item)
                temp_item_set = tuple(temp_item_set)
                if temp_item_set not in cnt_fr_list:
                    break
            else:
                unique_tokens.append(item_set)

        # new iteration
        if len(unique_tokens) == 0:
            break

    return cur_collection


if __name__ == "__main__":
    transactions = [
        ["l1", "l2", "l3", "l4", "l3"],
        ["l1", "l2", "l3", "l4" "l3", "l4", ],
        ["l1", "l2", "l3", "l4"],
        ["l1", "l2", "l3", "l4", "l2", "l2", "l3", "l5"],
        ["l1", "l2", "l3", "l4", "l1", "l2", "l3"],
        ["l1", "l2", "l3", "l4"]
    ]

    result = apriori(transactions, 0.5, -1)
    for k in result:
        print(k, ":")
        for v in result[k]:
            print("({})".format(' '.join(v)), end="  ")
        print()
