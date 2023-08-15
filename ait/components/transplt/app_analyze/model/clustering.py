import numpy as np
import os
import multiprocessing
import time
from functools import partial
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from dtaidistance import dtw, dtw_ndim

from app_analyze.model.train import Model
from app_analyze.model.embedding import api_embedding, seqs_embedding
from app_analyze.utils.log_util import logger


def jaccard_dist(a, b):
    union = np.union1d(a, b)
    intersection = np.intersect1d(a, b)
    dist = 1.0 - len(intersection) * 1.0 / len(union)
    return dist


# 进程池initializer函数
def init_pool(array):
    global glob_array  # 共享全局变量
    glob_array = array


# 子进程函数
def process_fn(ij, func=None, array_width=None):
    i, j, ai, aj = ij
    # 子进程读取全局变量glob_array，对齐一维glob_array与原始二维array的对应位置关系
    glob_array[i * array_width + j] = func(ai, aj)


def calc_relation_mat(func, list1, list2=None, relation='dist'):
    len1 = len(list1)
    len2 = len1 if list2 is None else len(list2)
    array = np.zeros((len1, len2))
    # Pool.map仅支持一个入参，使用偏函数functools.partial，预先传入其他参数
    fn_partial = partial(process_fn, func=func, array_width=array.shape[0])
    # array为展平的矩阵（即multiprocessing.RawArray, 多进程不支持二维矩阵）
    array_shared = multiprocessing.RawArray('d', array.ravel())
    # 由于各进程改动对应矩阵位置(即内存地址)处的值，无冲突，故无需加进程锁
    # 定义进程池，指定进程数量（processes），初始化函数（initializer）及其参数（initargs）
    n_proc = max(os.cpu_count(), 16)
    p = multiprocessing.Pool(processes=n_proc, initializer=init_pool, initargs=(array_shared,))
    # 若list1==list2，先计算下三角矩阵，然后转置后复值到上三角位置，否则全部计算
    if list2 is None:
        it = [(i, j, list1[i], list1[j]) for i in range(len1) for j in range(i)]
    else:
        it = [(i, j, list1[i], list2[j]) for i in range(len1) for j in range(len2)]
    # map函数向子进程函数分配不同的参数
    p.map(fn_partial, it)
    p.close()
    p.join()
    # glob_array为子进程中的全局变量，在主进程中并未被定义，主进程中的array_shared与子进程中的glob_array指向同一内存地址
    array = np.frombuffer(array_shared, np.double).reshape(array.shape)
    if list2 is None:
        if relation == 'dist':
            # 无需 - np.diag(np.diag(dist_mat))，因为对角线为0
            array = array + array.T
        elif relation == 'sim':
            # 对角线为1
            array = array + array.T + np.eye(len(list1))

    return array


def try_dbscan(embed=False):
    tik = time.time()
    model = Model()
    api_corpus, api_seqs, idx_seq_dict = model.train(seqs='./opencv.seqs.bin', seqs_idx='./opencv.seqs_idx.bin')
    api_seqs = list(api_seqs)  # [:50]
    tik1 = time.time()
    logger.info(f"1. time: {tik1 - tik}")

    if not embed:
        ##### Jaccard距离 + 无API Embedding
        dist_mat = calc_relation_mat(jaccard_dist, api_seqs, relation='dist')
        # dist_mat = calc_matrix(jaccard_dist, api_seqs)
        tik1 = time.time()
        logger.info(f"2. time: {tik1 - tik}")
        clustering = DBSCAN(eps=0.6, min_samples=2, metric='precomputed').fit(dist_mat)
        # clustering = AgglomerativeClustering(30).fit(dist_mat)
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
        dist_mat = calc_relation_mat(dtw_ndim.distance, embedding, relation='dist')
        clustering = DBSCAN(eps=0.6, min_samples=2, metric='precomputed').fit(dist_mat)
        tik1 = time.time()
        logger.info(f"4. time: {tik1 - tik}")

    logger.info([f'{i}:{x}' for i, x in enumerate(clustering.labels_) if x >= 0])
    logger.info([f'{api_seqs[i]}:{x}' for i, x in enumerate(clustering.labels_) if x >= 0])
    debug_string(clustering.labels_, idx_seq_dict, api_seqs)


def debug_string(labels, idx_seq_dict, api_seqs):
    import re

    name_rst = dict()
    id_rst = dict()
    for i, x in enumerate(labels):
        if not name_rst.get(x, None):
            name_rst[x] = []
            id_rst[x] = []
        ext = re.sub(r'\(.*?\)', '', '-->'.join([idx_seq_dict[_] for _ in api_seqs[i]]))
        name_rst[x].append(ext)
        id_rst[x].append(','.join([str(_) for _ in api_seqs[i]]))

    for key, val in name_rst.items():
        # name_str = '\n'.join(val)
        # id_str = '\n'.join(id_rst[key])
        name_str = ''
        for i in range(len(val)):
            name_str += f'{val[i]}\n'
            name_str += f'{id_rst[key][i]}\n'
        logger.info(f'{key}:\n{name_str}')


if __name__ == '__main__':
    try_dbscan(embed=False)
