# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-18
# @Description: 利用BLIP模型进行图文问答

from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
import numpy as np
import json
from utils import data_path, model_path, images_path

if __name__ == '__main__':
    # 加载数据
    img = Image.open(images_path.joinpath("scqxwrymypdzdefummyj.jpg")).convert('RGB')
    # question = "What color is the girl's hair in the picture?"
    question = "How many people are in the picture?"

    # 加载模型
    model = BlipForQuestionAnswering.from_pretrained(
        model_path.joinpath("blip-vqa-base"))
    processor = AutoProcessor.from_pretrained(
        model_path.joinpath("blip-vqa-base"))

    # 进行图文问答
    inputs = processor(img, question, return_tensors="pt")
    outputs = model.generate(**inputs)
    # print(outputs[0])
    print(processor.decode(outputs[0], skip_special_tokens=True))
