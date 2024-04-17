# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Author: ymh
# @Date: 2024-04-17
# @Description: 通过大模型对问题进行分类（图片描述类、细节查询类、细节判断类、图文匹配类）

import json
import time
import jwt
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from utils import data_path, ZHIPUAI_API_KEY


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


if __name__ == '__main__':
    with open(data_path.joinpath('query.json'), 'r', encoding='utf-8') as f:
        query = json.load(f)

    questions = [item['question'] for item in query]  # 提取问题数据

    # 定义大模型（此处是基于chatglm 3大模型）
    llm = ChatOpenAI(
        # model_name="glm-4",
        model_name="glm-3-turbo",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=generate_token(ZHIPUAI_API_KEY, exp_seconds=10000),
        streaming=False,
        temperature=0.1
    )

    # 构建提示工程，输出链
    system_prompt = """你是一个专业的问题分类工具。你需要理解问题的含义，对问题进行分类。
    问题的类型必须包含在[“图片描述类”，“细节查询类”，“细节判断类”，“图文匹配类”]中，具体示例如下：
    
    问题：请匹配到与 快鱼AGW系列T恤男新款宽松上衣海浪字母印花短袖T恤 最相关的图片。
    输出：图文匹配类
    
    问题：请对给定的图片进行描述。
    输出：图片描述类
    
    问题：这款运动裤是什么材质做的？
    输出：细节查询类
    
    问题：这件短款T恤是哪一年的新款？
    输出：细节查询类
    
    问题：这款衬衫是2019年的新款吗？
    输出：细节判断类
    
    问题：这件衣服是棉麻材质的吗？
    输出：细节判断类
    
    现在开始，根据下面的问题内容，严格按照示例信息的输出格式对问题进行分类：
    问题：{question}
    输出："""
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = {'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()

    # question = "这款牛仔裤的材质是什么？"
    # output = chain.invoke(question)
    # print(output)

    # 利用大模型对问题进行分类
    labels = []
    for q in questions:
        res = chain.invoke(q)
        labels.append(res)

    # 构建结果集并保存
    df = pd.DataFrame({'question': questions, 'label': labels})
    df.to_csv(data_path.joinpath('question_classify_result_llm.csv'), index=False)
    print(df)
