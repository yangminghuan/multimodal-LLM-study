# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-17
# @Description: 基于中文CLIP模型进行图文检索与匹配

import torch
import numpy as np
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.image import load_image
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
    img = load_image(str(images_path.joinpath("scqxwrymypdzdefummyj.jpg")))

    # 加载模型
    pipeline = pipeline(task=Tasks.multi_modal_embedding,
                        model=str(model_path.joinpath("multi-modal_clip-vit-large-patch14_zh")),
                        model_revision='v1.0.1')

    # 支持一张图片(PIL.Image)或多张图片(List[PIL.Image])输入，输出归一化特征向量
    img_embedding = pipeline.forward({'img': img})[
        'img_embedding']  # 2D Tensor, [图片数, 特征维度]

    # 支持一条文本(str)或多条文本(List[str])输入，输出归一化特征向量
    text_embedding = pipeline.forward({'text': query_texts})[
        'text_embedding']  # 2D Tensor, [文本数, 特征维度]

    # 模型推理，计算图文相似度
    with torch.no_grad():
        # 计算内积得到logit，考虑模型temperature
        logits_per_image = (img_embedding / pipeline.model.temperature) @ text_embedding.t()
        # 根据logit计算概率分布
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("图文匹配概率:", np.around(probs, 3))
