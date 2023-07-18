from prefixspan import PrefixSpan


# 加载数据集
def load_dataset():
    # db = [
    #     [0, 1, 2, 3, 4],
    #     [1, 1, 1, 3, 4],
    #     [2, 1, 2, 2, 0],
    #     [1, 1, 1, 2, 2],
    # ]

    # dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    # dataset = [
    #     [0, 1, 0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 7, 9, 10, 11, 12, 13, 11, 9, 14, 8, 15, 8, 16, 17, 18, 19, 20, 21, 13,
    #      11, 13, 11, 13, 11, 13, 11, 22, 13, 23]]

    # dataset = [[0, 1, 0, 1], [0, 1, 2, 3, 4], [3, 4, 5, 6, 5, 6], [5, 6, 7, 8, 7, 9, 10],
    #            [11, 12, 13, 11, 9, 14], [9, 14, 8, 15, 8, 16, 17, 18, 19, 20, 21], [13, 11, 13, 11, 13, 11, 13, 11],
    #            [13, 11, 22, 13, 23]]

    dataset = [[0, 1, ], [5, 6], [13, 11],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 10, 11, 12, 13, 11, 9, 14, 8, 15, 8, 16, 17, 18, 19, 20, 21, 13,
                11, 22, 13, 23]]

    return dataset


def prefixspan(dataset, top_k=5, frq=2):
    ps = PrefixSpan(dataset)

    freq = ps.frequent(frq, closed=True)
    # l1 = ps.topk(top_k, closed=True, filter=lambda patt, matches: matches[0][0] > 0)
    l1 = ps.topk(top_k, closed=True)

    # print(ps.frequent(frq, generator=True))
    # print(ps.topk(top_k, generator=True))

    # print(f'Top({top_k}): ', l1)
    # print(f'frequent({frq}): ', freq)
    return l1, freq


def test_prefixspan():
    dataset = load_dataset()
    print('dataSet: ', dataset)

    prefixspan(dataset)


if __name__ == "__main__":
    # 测试 Apriori 算法
    test_prefixspan()
