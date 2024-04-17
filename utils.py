# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-16
# @Description: 定义常用的变量和工具函数

import os
from pathlib import Path

root_path = Path(__file__).parent
data_path = root_path.joinpath('dataset')
model_path = root_path.joinpath('models')
images_path = data_path.joinpath('images')

ZHIPUAI_API_KEY = ""  # 此处填写自己申请的智谱AI的key

# if __name__ == '__main__':
#     import requests
#     import base64
#
#     url = "http://34.143.180.202:3389/viscpm"
#     resp = requests.post(url, json={
#         # need to modify
#         "image": base64.b64encode(open(images_path.joinpath("scqxwrymypdzdefummyj.jpg"), "rb").read()).decode(),
#         "question": "描述一下这张图片",
#     })
#     resp = resp.json()
#     print(resp)
