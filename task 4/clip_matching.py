# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-17
# @Description: 基于CLIP模型进行图文检索和匹配

from PIL import Image
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json
from utils import data_path, model_path, images_path

if __name__ == '__main__':
    # 加载问题数据
    # with open(data_path.joinpath('query.json'), 'r', encoding='utf-8') as f:
    #     query = json.load(f)
    # query_texts = [item['question'] for item in query]
    query_texts = ["裙子女2019新款夏季气质时尚女装中长款衬衫裙收腰显瘦polo连衣裙",
                   "海尼2019新款仙女连衣裙碎花气质显瘦衬衫裙收腰短袖中长裙子女夏",
                   "ON＆ON/安乃安商场同款夏季韩版优雅休闲纯色圆领格纹短袖衬衫女",
                   "2019夏季新款韩版气质连衣裙定制法式复古格纹小众连衣裙女过膝裙",
                   "无拘2019女夏新款衬衫裙夏装格纹收腰气质显瘦蕾丝腰带衬衫连衣裙",
                   "影儿恩裳2019夏季新款荷叶v领前襟短袖收腰桑蚕丝真丝连衣裙",
                   "影儿诗篇女装2019夏季新款粉蓝修身收腰拼接网纱蕾丝连衣裙"]

    # 加载Taiyi 中文 text encoder
    text_tokenizer = BertTokenizer.from_pretrained(
        model_path.joinpath("Taiyi-CLIP-Roberta-102M-Chinese"))
    text_encoder = BertForSequenceClassification.from_pretrained(
        model_path.joinpath("Taiyi-CLIP-Roberta-102M-Chinese")).eval()
    text = text_tokenizer(query_texts, return_tensors='pt', padding=True)[
        'input_ids']

    # 加载CLIP的image encoder
    img_path = images_path.joinpath("scqxwrymypdzdefummyj.jpg")  # 加载图片路径
    clip_model = CLIPModel.from_pretrained(
        model_path.joinpath("clip-vit-base-patch32"))
    processor = CLIPProcessor.from_pretrained(
        model_path.joinpath("clip-vit-base-patch32"))
    image = processor(images=Image.open(img_path), return_tensors="pt")

    # 计算图文相似度
    with torch.no_grad():
        # 特征提取
        image_features = clip_model.get_image_features(**image)
        text_features = text_encoder(text).logits
        # 归一化
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # 计算余弦相似度 logit_scale是尺度系数
        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(np.around(probs, 3))
        print(np.argmax(probs))
