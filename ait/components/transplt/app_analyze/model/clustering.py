import numpy as np
import time
from sklearn.cluster import DBSCAN
from dtaidistance import dtw, dtw_ndim

from app_analyze.model.train import Model
from app_analyze.model.embedding import api_embedding, seqs_embedding
from app_analyze.utils.log_util import logger


def jaccard_dist(a, b):
    union = np.union1d(a, b)
    intersection = np.intersect1d(a, b)
    dist = 1.0 - len(intersection) * 1.0 / len(union)
    return dist


def calc_dist_mat(func, list1, list2=None):
    if not list2:
        list2 = list1
    dist_mat = np.ones((len(list1), len(list2))) * np.inf
    for i, acc_api in enumerate(list1):
        for j, asd_api in enumerate(list2):
            dist_mat[i, j] = func(acc_api, asd_api)
    return dist_mat


def try_dbscan(embed=False):
    tik = time.time()
    model = Model()
    api_corpus, api_seqs = model.train(seqs='./seqs.bin', seqs_idx='./seqs_idx.bin')
    api_seqs = list(api_seqs)[:50]
    tik1 = time.time()
    logger.info(f"1. time: {tik1 - tik}")

    if not embed:
        ##### Jaccard距离 + 无API Embedding
        dist_mat = calc_dist_mat(jaccard_dist, api_seqs)
        tik1 = time.time()
        logger.info(f"2. time: {tik1 - tik}")
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='precomputed').fit(dist_mat)
        tik1 = time.time()
        logger.info(f"3. time: {tik1 - tik}")
    else:
        ##### API Embedding + DTW距离
        # 用原始API序列学习API Embedding
        corpus = []
        for seq in api_corpus:
            seq_ = list()
            if isinstance(seq, list):
                seq_ = seq  # 直接对ID做Embedding
            else:
                for x in seq.api_seq:
                    seq_.append(x.api_name)
            corpus.append(seq_)
        api2vec = api_embedding(corpus, vector_size=10, window=5, min_count=1, workers=4)
        tik1 = time.time()
        logger.info(f"2. time: {tik1 - tik}")

        # 对高频API序列向量化
        embedding = seqs_embedding(api2vec, api_seqs)
        tik1 = time.time()
        logger.info(f"3. time: {tik1 - tik}")

        # 非padding模式，不等长。dtw参考：https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html
        dist_mat = calc_dist_mat(dtw_ndim.distance, embedding)
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='precomputed').fit(dist_mat)
        tik1 = time.time()
        logger.info(f"4. time: {tik1 - tik}")

    logger.info([f'{i}:{x}' for i, x in enumerate(clustering.labels_) if x >= 0])
    logger.info([f'{api_seqs[i]}:{x}' for i, x in enumerate(clustering.labels_) if x >= 0])


if __name__ == '__main__':
    try_dbscan(embed=False)
