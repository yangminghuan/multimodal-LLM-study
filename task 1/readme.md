## 任务1：读取数据集
- 任务说明：读取比赛数据数据，了解问答数据集
- 任务要求：
    - 了解数据集背景
    - 读取比赛数据集
    - 阅读问答技术背景

比赛数据集的设计是为了模拟真实的电商环境，让参赛者能够开发出能够理解和处理实际场景中图像与文本信息的算法模型。训练集包含了原始的图片和文本描述，用于参赛者训练他们的算法模型。通过训练集，模型可以学习如何从图像和文本中提取特征，并建立起图像内容与文本描述之间的关联。

测试集为模拟生成的用户提问，需要参赛选手结合提问(question)和提问图片(related_image)进行生成回答(answer)，其中测试集的样例格式如下：
```
 {
  "question": "这套衣服是什么材质的？",
  "related_image": "xkbxadltjkifbgutksss.jpg",
  "answer": ""
 },
 {
  "question": "请匹配到与 快鱼AGW系列T恤男新款宽松上衣海浪字母印花短袖T恤 最相关的图片。",
  "related_image": "",
  "answer": ""
 },
...
```
在提问中问题存在以下几种类型：

- 通过提问(question)对提问图片(related_image)进行提问和描述
- 通过提问(question)检索到最相关的图片(related_image)

多模态数据集是指包含多种不同类型数据的数据集合，在多模态数据集中，每种类型的数据都提供了关于数据集主题的不同视角和信息。这种数据集在现实世界的应用中非常广泛，因为现实世界的信息往往是以多种不同形式存在的。

多模态问答（Multimodal Question Answering, MQA）是一种人工智能任务，它结合了来自不同模态的信息，如文本、图像、音频和视频，以提供更准确和全面的答案。多模态问答系统通常具有交互性，允许用户通过不同的方式提问，例如使用自然语言、点击图片中的具体区域或提供音频输入。

多模态问答的挑战：

1. 数据融合技术：如何有效地融合来自不同模态的数据是一个主要挑战。
2. 特征提取：从多模态数据中提取相关特征需要特定的技术和方法。
3. 语义理解：系统需要具备强大的语义理解能力，以便从文本、图像和其他模态中提取深层次的意义，并将其与问题相关联。