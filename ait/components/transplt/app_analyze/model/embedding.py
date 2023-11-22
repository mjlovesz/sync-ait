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

import numpy as np
from gensim.models import Word2Vec
from dtaidistance import dtw, dtw_ndim
from sklearn.cluster import DBSCAN

from app_analyze.utils.log_util import logger


def one_hot_embedding(labels, num_classes):
    """
    将标签转换为one-hot编码

    Args:
        labels: 标签列表，形状为（样本数，）
        num_classes: 类别数

    Return:
        one_hot: one-hot编码的标签，形状为（样本数，类别数）
    """
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot


def api_embedding(corpus, vector_size=10, window=5, min_count=1, workers=4):
    model = Word2Vec(corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model.bin')
    model = Word2Vec.load('model.bin')
    return model


def seq_embedding(model, seq, max_len=None):
    if not seq:
        return None
    seq_embed = list()
    for vocab in seq:
        seq_embed.append(model.wv[vocab])
    if not max_len:
        return np.array(seq_embed)
    pad_len = max_len - len(seq)
    # 或可使用：seq_embed = np.pad(seq_embed, ((0,0), (0, pad_len)), mode='constant')
    if pad_len > 0:
        vector_size = seq_embed[-1].shape[0]
        for _ in range(pad_len):
            seq_embed.append(np.zeros((vector_size,)))
    return np.array(seq_embed)


def seqs_embedding(model, api_seqs, max_len=None):
    # 对高频API序列向量化
    min_len = np.inf
    max_len = max_len or 0
    for seq in api_seqs:
        max_len = max(max_len, len(seq))
        min_len = min(min_len, len(seq))
    logger.debug(f'seq len [{min_len}, {max_len}]')
    embedding = list()
    for seq in api_seqs:
        embed = seq_embedding(model, seq, max_len=max_len)
        embedding.append(embed)
    return embedding


def calc_sim_mat(func, list1, list2=None):
    if not list2:
        list2 = list1
    mat = np.zeros((len(list1), len(list2)))
    for i, acc_api in enumerate(list1):
        for j, asd_api in enumerate(list2):
            mat[i, j] = func(acc_api, asd_api)
    return mat


def calc_dist_mat(func, list1, list2=None):
    if not list2:
        list2 = list1
    mat = np.ones((len(list1), len(list2))) * np.inf
    for i, acc_api in enumerate(list1):
        for j, asd_api in enumerate(list2):
            mat[i, j] = func(acc_api, asd_api)
    return mat


def mat_cos_similarity(mat1, mat2, eps=1e-5):
    norm1 = np.linalg.norm(mat1, axis=-1, keepdims=True)
    norm2 = np.linalg.norm(mat2, axis=-1, keepdims=True)
    mat1 = mat1 / (norm1 + eps)
    mat2 = mat2 / (norm2 + eps)
    cos = np.dot(mat1, mat2.T)
    cos = np.sum(cos) / np.count_nonzero(cos)
    return cos


def mat_dtw_similarity(mat1, mat2):
    """https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html"""
    return 1 - dtw_ndim.distance(mat1, mat2)


def try_gensim_embedding():
    """
    需要numpy>1.21.2，目前看1.24.2可用。

    https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
    """
    # 1. 收集语料库
    acc_corpus = [
        # NV
        ['imdecode', 'crop', 'resize', 'normalize', 'Image'],
        ['imread', 'imdecode', 'Rect', 'Mat', 'resize', 'normalize', 'Mat'],
        ['imread', 'imdecode', 'size', 'Rect', 'Mat', 'resize', 'divide', 'Mat'],
        ['imread', 'imdecode', 'copyTo', 'Rect', 'Mat', 'resize', 'normalize', 'Mat'],
        ['imread', 'imdecode', 'Rect', 'Mat', 'resize', 'normalize', 'imencode'],
    ]

    asd_corpus = [
        # 昇腾：Decode=imdecode，Encode=imencode
        ['Decode', 'crop', 'resize', 'normalize', 'Image'],
        ['UNK', 'Decode', 'crop', 'resize', 'normalize', 'Image'],
        ['Decode', 'GetSize', 'crop', 'resize', 'divide', 'Image'],
        ['Decode', 'copyTo', 'crop', 'resize', 'normalize', 'Image'],
        ['Decode', 'crop', 'resize', 'normalize', 'Encode'],
    ]

    # 基于专家映射表，针对昇腾API预料进行等价API替换，目的是便于将NV和昇腾API向量关联到一起，否则两边学到的API向量就无关了。
    api_map = {'Decode': 'imdecode', 'Encode': ''}
    for seq in asd_corpus:
        for i, api in enumerate(seq):
            seq[i] = api_map.get(api, api)

    corpus = acc_corpus + asd_corpus

    # 2. 训练词嵌入模型
    vector_size = 10
    model = Word2Vec(corpus, vector_size=vector_size, window=5, min_count=1, workers=4)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model.bin')
    model = Word2Vec.load('model.bin')

    # 3.
    embedding = model.wv['Rect']
    logger.debug(embedding)
    # 例如：('Rect', 'crop') ('Rect', 'normalize')
    logger.debug(model.wv.similarity('Mat', 'Image'))
    logger.debug(model.wv.most_similar(positive=['Rect', 'crop'], topn=2))

    # API序列向量化
    max_len = 8
    acc_embedding = list()
    asd_embedding = list()
    for seq in acc_corpus:
        acc_embedding.append(seq_embedding(model, seq, max_len, vector_size))
    for seq in asd_corpus:
        asd_embedding.append(seq_embedding(model, seq, max_len, vector_size))

    # 打印相似性矩阵
    seq_sim_mat = calc_sim_mat(mat_dtw_similarity, acc_embedding, asd_embedding)
    logger.debug(seq_sim_mat)

    # Sklearn DBSCAN仅支持2维数据集，即API序列必须为1维
    acc_embedding = [x.flatten() for x in acc_embedding]
    asd_embedding = [x.flatten() for x in asd_embedding]
    seq_sim_mat = calc_sim_mat(dtw.distance, acc_embedding, asd_embedding)
    logger.debug(seq_sim_mat)
    clustering = DBSCAN(eps=0.2, min_samples=2, metric=dtw.distance).fit(acc_embedding + asd_embedding)
    logger.debug(clustering.labels_)
    for i, x in enumerate(acc_corpus + asd_corpus):
        logger.debug(clustering.labels_[i], x)


if __name__ == '__main__':
    try_gensim_embedding()
