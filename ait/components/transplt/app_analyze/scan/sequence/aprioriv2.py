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

from __future__ import print_function
from numpy import *


# 加载数据集
def load_dataset():
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    return dataset


# 创建集合 C1。即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
def create_c1(dataset):
    """create_c1（创建集合 c1）

    Args:
        dataset 原始数据集
    Returns:
        frozenset 返回一个 frozenset 格式的 list
    """

    c1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                # 遍历所有的元素，如果不在 C1 出现过，那么就 append
                c1.append([item])

    # 对数组进行 `从小到大` 的排序
    c1.sort()
    # frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用
    return list(map(frozenset, c1))


# 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
def scan_dataset(dataset, ck, min_support):
    """scan_dataset（计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度 minSupport 的数据）

    Args:
        dataset 数据集
        ck 候选项集列表
        min_support 最小支持度
    Returns:
        ret_list 支持度大于 minSupport 的集合
        support_data 候选项集支持度数据
    """

    # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8
    ss_cnt = {}
    for tid in dataset:
        for can in ck:
            # s.issubset(t)  测试是否 s 中的每一个元素都在 t 中
            if not can.issubset(tid):
                continue

            if not ss_cnt.get(can):
                ss_cnt[can] = 1
            else:
                ss_cnt[can] += 1

    num_items = float(len(dataset))  # 数据集 D 的数量
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ss_cnt[key] / num_items
        if support >= min_support:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            ret_list.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        support_data[key] = support
    return ret_list, support_data


# 输入频繁项集列表 lk 与返回的元素个数 k，然后输出所有可能的候选项集 ck
def apriori_gen(lk, k):
    """apriori_gen（输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck。
       例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
       仅需要计算一次，不需要将所有的结果计算出来，然后进行去重操作
       这是一个更高效的算法）

    Args:
        lk 频繁项集列表
        k 返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
    Returns:
        retList 元素两两合并的数据集
    """

    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):
            l1 = list(lk[i])[: k - 2]
            l2 = list(lk[j])[: k - 2]
            l1.sort()
            l2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            # if first k-2 elements are equal
            if l1 == l2:
                # set union
                ret_list.append(lk[i] | lk[j])
    return ret_list


# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataset, min_support=0.5):
    """apriori（首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，C2 再进一步过滤变成 L2，然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。）

    Args:
        dataset 原始数据集
        min_support 支持度的阈值
    Returns:
        L 频繁项集的全集
        supportData 所有元素和支持度的全集
    """
    # c1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    c1 = create_c1(dataset)
    # 对每一行进行 set 转换，然后存放到集合中
    d = list(map(set, dataset))
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    l1, support_data = scan_dataset(d, c1, min_support)

    # L 加了一层 list, L 一共 2 层 list
    l = [l1]
    k = 2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while len(l[k - 2]) > 0:
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        ck = apriori_gen(l[k - 2], k)

        lk, sup_k = scan_dataset(d, ck, min_support)  # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        # 保存所有候选项集的支持度，如果字典没有，就追加元素，如果有，就更新元素
        support_data.update(sup_k)
        if len(lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        l.append(lk)
        k += 1
    return l, support_data


def test_apriori():
    # 加载测试数据集
    dataset = load_dataset()

    # Apriori 算法生成频繁项集以及它们的支持度
    l1, support_data1 = apriori(dataset, min_support=0.7)
    # Apriori 算法生成频繁项集以及它们的支持度
    l2, support_data2 = apriori(dataset, min_support=0.5)


if __name__ == "__main__":
    # 测试 Apriori 算法
    test_apriori()
