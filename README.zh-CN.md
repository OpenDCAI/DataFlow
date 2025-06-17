# DataFlow

<div align="center">
  <img src="./static/images/Face.png">

[![](https://img.shields.io/github/license/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/Open-DataFlow/DataFlow?style=social)](https://github.com/Open-DataFlow/DataFlow)
[![](https://img.shields.io/github/issues-raw/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/issues)
[![](https://img.shields.io/github/last-commit/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/Data/Flowcommits/main/)
[![](https://img.shields.io/github/contributors/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/graphs/contributors)

[简体中文](./README.zh-CN.md) | English

**[特性](#特性) • [快速开始](#快速开始) • [文档](https://open-dataflow.github.io/DataFlow-Doc/) • [贡献](#贡献) • [许可证](#许可证)**

</div>

## 新闻
- [2025-07-25] 🎉 我们发布了 Dataflow-agent。
- [2025-06-30] 🎉 我们发布了 Dataflow 的文档。
- [2024-10-14] 🎉 我们在 [👋 Awesome_Data_Evaluation](./Awesome_Data_Evaluation.md) 中总结了数据评估相关的论文和代码。
- [2024-10-14] 🎉 我们的第一个以数据为中心的评估系统现已开源。

## 概述
DataFlow 是一个数据评估和处理系统，旨在从嘈杂的数据源（如 PDF、纯文本、低质量问答）中 **清洗、扩增和评估** 高质量训练数据，从而通过针对性训练（预训练、监督微调、强化学习训练）提升大型语言模型（LLM, large language model）在特定领域的表现。**DataFlow已经在医疗、金融和法律等领域通过实验证明可以提升面向领域的大模型性能。**

具体来说，我们构建了多样化的 `算子`（operator），利用基于规则的方法、深度学习模型、大语言模型（LLMs）和 LLM API。这些算子被系统地集成到六个不同的 `流水线`（Pipeline） 中，共同构成了完整的 `Dataflow` 系统。此外，我们还开发了一个智能 `Agent`，能够根据任务需求动态组合现有的 `算子`，自动构建新的 `Pipeline`。

## 管道与代理
Dataflow 当前的管道如下：
- **Text Pipeline**：从大规模纯文本数据中挖掘问答对，用于 SFT 和强化学习训练。
- **Reasoning Pipeline**：对现有的问答对进行增强，包括（1）扩展思维链（COT），（2）分类，（3）难度估计。
- **Text2SQL Pipeline**：将自然语言问题翻译成 SQL 查询，并补充解释、思维链和上下文模式信息。
- **Agentic RAG Pipeline**：从现有的问答数据集或知识库中识别并提取需要外部知识来回答的问答对，用于下游的Agentic RAG 任务训练。
- **知识库清洗管道**：从表格、PDF 和 Word 文档等数据来源中提取并结构化知识，生成可用于下游 RAG 或问答对生成的条目。

在此基础上，我们还提供了 **DataFlow Agent**，可以根据任务需求安排现有的 `operator` 并自动构建新的管道。

## 快速开始
对于环境设置和安装，请使用以下命令👇

```shell
conda create -n dataflow python=3.10
conda activate dataflow

git clone https://github.com/Open-DataFlow/DataFlow
cd DataFlow
pip install -e .
```

对于 **快速开始** 和 **指南**，请访问我们的 [文档](https://open-dataflow.github.io/DataFlow-Doc/)。

## 特性与可视化

### 1. 文本管道

### 2. 推理管道
![](./static/images/demo_reasoning.png)

您可以参考我们在 Huggingface 上的 [推理管道样本](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning/)，查看演示输入和输出。

- 性能提升：
  - ![](./static/images/reasoning_performance.png)

### 3. 文本转 SQL 管道

## 引用
```plaintext
@article{wang2025rare,
  title={Rare: Retrieval-augmented reasoning modeling},
  author={Wang, Zhengren and Yu, Jiayang and Ma, Dongsheng and Chen, Zhe and Wang, Yu and Li, Zhiyu and Xiong, Feiyu and Wang, Yanfeng and Tang, Linpeng and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2503.23513},
  year={2025}
}
```

## 统计信息
<a href="https://star-history.com/#Open-DataFlow/DataFlow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
 </picture>
</a>
