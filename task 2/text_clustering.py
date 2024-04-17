# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-16
# @Description: 通过文本聚类对问题进行分类

import json
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from utils import data_path

jieba.initialize()

if __name__ == '__main__':
    with open(data_path.joinpath('query.json'), 'r', encoding='utf-8') as f:
        query = json.load(f)

    # 基于jieba库对问题文本进行分词，构建语料库
    questions = [item['question'] for item in query]
    # print(questions)
    corpus = [' '.join(jieba.lcut(q)) for q in questions]  # 利用空格进行分隔
    # print(corpus)

    # 利用词频提取文本特征，构建文本特征矩阵
    # count_vec = CountVectorizer()
    # X = count_vec.fit_transform(corpus)
    # # print(X.toarray())
    # cols = ['fea_' + str(i) for i in range(X.toarray().shape[1])]
    # # print(cols)
    # cluster_data = pd.DataFrame(data=X.toarray(), columns=cols)
    # # print(cluster_data.head())

    # 利用TF-IDF提取文本特征
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus)
    # print(X.toarray())
    cols = ['fea_' + str(i) for i in range(X.shape[1])]
    cluster_data = pd.DataFrame(data=X.toarray(), columns=cols)
    # print(cluster_data.head())

    # 定义k-means聚类模型，对问题文本进行聚类分组
    k_means = KMeans(n_clusters=4, n_init='auto', random_state=2024)
    k_means.fit(cluster_data)
    # print(k_means.labels_)
    cluster_data['question'] = questions
    cluster_data['label'] = k_means.labels_
    print(cluster_data.label.value_counts())
    print(cluster_data[cluster_data.label == 3]['question'].values)
