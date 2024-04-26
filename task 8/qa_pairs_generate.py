# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-24
# @Description: 基于GLM-4大模型生成问答对数据集

import json
import pandas as pd
from langchain_community.chat_models import ChatOpenAI, ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import data_path, ZHIPUAI_API_KEY, generate_token


if __name__ == '__main__':
    train_data = pd.read_csv(data_path.joinpath("train_annotation.csv"), sep='\t')
    desc_qa_pairs = []
    for _, row in train_data.iterrows():
        desc_qa_pairs.append(['图片描述类', '请对给定的图片进行描述。', row[0], row[1]])
    # print(desc_qa_pairs[:10])
    desc_qa_pairs_df = pd.DataFrame(desc_qa_pairs, columns=['label', 'question', 'related_image', 'answer'])
    desc_qa_pairs_df.to_csv(data_path.joinpath('desc_qa_pairs_df.csv'), index=False)

    question_classify_df = pd.read_csv(data_path.joinpath('question_classify_result_llm.csv'))
    query_classify = question_classify_df[question_classify_df.label == '细节查询类']
    # print(query_classify.question.value_counts()[:20])

    judge_classify = question_classify_df[question_classify_df.label == '细节判断类']
    # print(judge_classify.question.value_counts()[:20])

    texts = train_data['text'].to_list()

    # 定义大模型（此处是基于glm-4大模型）
    # llm = ChatZhipuAI(
    #     model_name="glm-4",
    #     # model_name="glm-3-turbo",
    #     api_key=ZHIPUAI_API_KEY,
    #     streaming=False,
    #     temperature=0.1
    # )
    llm = ChatOpenAI(
        model_name="glm-4",
        # model_name="glm-3-turbo",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=generate_token(ZHIPUAI_API_KEY, exp_seconds=3000),
        streaming=False,
        temperature=0.1
    )
    # print(llm.invoke("你是谁？"))
    # """You are given a scientific article and a question. Answer the question as conciscly as you can, using a single phrase or
    # sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If
    # the question is a yes/no question, answer "yes","no" or "unanswerable". Do not provide any explanation."""

    # system_prompt = """你是一个问答对数据集处理专家。你的任务是根据提供的文本内容，生成对应的问答对。
    # 问题示例集合：['这款连衣裙是什么风格的？', '这款裙子是2019年的新款吗？', '这款连衣裙是什么季节穿的？', '这款T恤是短袖的吗？', '这款连衣裙是什么材质的？', '这款连衣裙是什么颜色的？', '这款连衣裙是什么品牌的？', '这套衣服的面料是什么？', '这条裤子是韩版的吗？']
    #
    # 问答对生成步骤如下：
    # 1.仔细阅读文本内容，提出合理的问题，要求提出的问题可以在文本中找到对应的答案，且问题的类型尽可能与问题示例集合保持一致；
    # 2.基于步骤1提出的问题，结合文本内容简要地生成答案；
    # 3.综合上述步骤的结果，以列表格式依次输出文本内容、提出的问题和生成的答案。
    #
    # 示例如下：
    # 文本内容：无拘2019女夏新款衬衫裙夏装格纹收腰气质显瘦蕾丝腰带衬衫连衣裙
    # 输出：['无拘2019女夏新款衬衫裙夏装格纹收腰气质显瘦蕾丝腰带衬衫连衣裙', '这款连衣裙是什么季节穿的？', '夏季']
    #
    # 现在开始，根据下面给出的文本内容，务必遵循上述的生成步骤进行推理，严格按照示例信息的格式输出结果：
    # 文本内容：{content}
    # 输出："""
    system_prompt = """你是一个问答对数据集处理专家。你的任务是根据提供的文本内容，生成对应的问答对。
    
    问答对生成步骤如下：
    1.仔细阅读文本内容，提出合理的问题，要求提出的问题可以在文本中找到对应的答案；
    2.基于步骤1提出的问题，结合文本内容简要地生成答案；
    3.综合上述步骤的结果，以列表格式依次输出文本内容、提出的问题和生成的答案。

    示例信息如下：
    示例一：
    文本内容：无拘2019女夏新款衬衫裙夏装格纹收腰气质显瘦蕾丝腰带衬衫连衣裙
    输出：['无拘2019女夏新款衬衫裙夏装格纹收腰气质显瘦蕾丝腰带衬衫连衣裙', '这款连衣裙是什么季节穿的？', '夏季']
    示例二：
    文本内容：2019夏季新款高端气质不对称肩带chic修身显瘦日常V领连衣裙女潮
    输出：['2019夏季新款高端气质不对称肩带chic修身显瘦日常V领连衣裙女潮', '这款裙子是2019年的新款吗？', '是']

    现在开始，根据下面给出的文本内容，务必遵循上述的生成步骤进行推理，严格按照示例信息的格式输出结果，每个文本内容只需要生成一个问答对即可：
    文本内容：{content}
    输出："""
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = {'content': RunnablePassthrough()} | prompt | llm | StrOutputParser()

    # content = "Material Girl高腰破洞牛仔裤女宽松2020春季新款工装裤显瘦百搭"
    # output = chain.invoke(content)
    # print(output)

    llm_qa_pairs = []
    qa_path = data_path.joinpath('qa')
    for idx, row in train_data.iloc[:500].iterrows():
        img = row[0]
        content = row[1]
        output = eval(chain.invoke(content))
        if isinstance(output[0], list):
            for i in output:
                llm_qa_pairs.append(['其他类', i[1], img, i[2]])
        else:
            llm_qa_pairs.append(['其他类', output[1], img, output[2]])

        if (idx + 1) % 100 == 0:
            qa_df = pd.DataFrame(llm_qa_pairs, columns=['label', 'question', 'related_image', 'answer'])
            qa_df.to_csv(qa_path.joinpath(f'qa_pairs_df_{len(qa_df)}.csv'), index=False)

    qa_df = pd.DataFrame(llm_qa_pairs, columns=['label', 'question', 'related_image', 'answer'])
    qa_df.to_csv(qa_path.joinpath(f'qa_pairs_df_all.csv'), index=False)
