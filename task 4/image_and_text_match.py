# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-23
# @Description: 基于中文CLIP模型解决图文匹配类型的问题，生成结果并保存

import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import random
import json
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from utils import data_path, model_path, images_path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # 读取问题类别结果集，筛选图文匹配类型的问题
    question_classify_df = pd.read_csv(data_path.joinpath("question_classify_result_llm.csv"))
    match_question_df = question_classify_df[question_classify_df.label == '图文匹配类']
    # 读取训练数据集中已出现的图片文件名，将其排除
    train_df = pd.read_csv(data_path.joinpath("train_annotation.csv"), sep='\t')
    train_images_names = train_df['image'].to_list()
    images_file_names = os.listdir(images_path)
    test_images = list(set(images_file_names) - set(train_images_names))
    # 截取问题中的图片描述文本片段
    questions = match_question_df['question'].to_list()
    texts = [q[6:-8] for q in questions]

    # 加载模型
    model = ChineseCLIPModel.from_pretrained(
        model_path.joinpath("chinese-clip-vit-large-patch14"))
    processor = ChineseCLIPProcessor.from_pretrained(
        model_path.joinpath("chinese-clip-vit-large-patch14"))

    # 加载图片数据集
    images = [Image.open(images_path.joinpath(img)) for img in test_images]

    # compute image-text similarity scores
    inputs = processor(text=texts[:10], images=images[:10], return_tensors="pt",
                       padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).detach().numpy()
    # print("图文匹配概率:", np.around(probs, 3))
    # 遍历得到文本最相关的图片文件
    related_images = [test_images[i] for i in np.argmax(probs, axis=1)]
    # 生成答案结果
    results = []
    for q, img in zip(questions, related_images):
        tmp = {
            "question": q,
            "related_image": img,
            "answer": ""
        }
        results.append(tmp)

    # print(results)
    # 保存结果集为json格式
    with open(data_path.joinpath("match_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f)
