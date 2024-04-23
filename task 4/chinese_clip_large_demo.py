# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-23
# @Description: 基于中文CLIP模型进行图文检索与匹配（huggingface.co）

from PIL import Image
import numpy as np
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
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
    # 加载图片数据
    image = Image.open(images_path.joinpath("scqxwrymypdzdefummyj.jpg"))

    # 加载模型
    model = ChineseCLIPModel.from_pretrained(
        model_path.joinpath("chinese-clip-vit-large-patch14"))
    processor = ChineseCLIPProcessor.from_pretrained(
        model_path.joinpath("chinese-clip-vit-large-patch14"))

    # compute image feature
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1,
                                                          keepdim=True)  # normalize

    # compute text features
    inputs = processor(text=query_texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1,
                                                       keepdim=True)  # normalize

    # compute image-text similarity scores
    inputs = processor(text=query_texts, images=image, return_tensors="pt",
                       padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).detach().numpy()
    print("图文匹配概率:", np.around(probs, 3))
