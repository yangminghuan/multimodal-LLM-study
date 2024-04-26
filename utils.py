# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-16
# @Description: 定义常用的变量和工具函数

import os
import time
import jwt
from pathlib import Path

root_path = Path(__file__).parent
data_path = root_path.joinpath('dataset')
model_path = root_path.joinpath('models')
images_path = data_path.joinpath('images')

ZHIPUAI_API_KEY = ""  # 此处填写自己申请的智谱AI的key


def generate_token(apikey: str, exp_seconds: int):
    """
    生成token令牌
    :param apikey: 接口key
    :param exp_seconds: 令牌有效时间（秒）
    :return:
    """
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


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
