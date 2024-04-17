# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-16
# @Description: 定义常用的变量和工具函数

import os
from pathlib import Path

root_path = Path(__file__).parent
data_path = root_path.joinpath('dataset')


ZHIPUAI_API_KEY = ""  # 此处填写自己申请的智谱AI的key
