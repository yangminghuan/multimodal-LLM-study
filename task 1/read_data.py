# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-16
# @Description: 读取并查看数据集

import pandas as pd
import json
from utils import data_path


if __name__ == '__main__':
    train_data = pd.read_csv(data_path.joinpath('train_annotation.csv'), sep='\t')
    print(train_data.shape)
    with open(data_path.joinpath('query.json'), 'r', encoding='utf-8') as f:
        query = json.load(f)
    print(query[:3])
