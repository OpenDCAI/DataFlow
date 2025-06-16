# DataFlow

<div align="center">
  <img src="./static/images/Face.png">


[![](https://img.shields.io/github/license/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/Open-DataFlow/DataFlow?style=social)](https://github.com/Open-DataFlow/DataFlow)
[![](https://img.shields.io/github/issues-raw/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/issues)
[![](https://img.shields.io/github/last-commit/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/commits/main/)
[![](https://img.shields.io/github/contributors/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/graphs/contributors)

[简体中文](./README.zh-CN.md) | English


**[Features](#Features) • [Quick Start](#Quick_Start) • [Documentation](https://open-dataflow.github.io/DataFlow-Doc/) • [Contributing](#贡献) • [License](#许可证)**


</div>


## News
- [2025-07-25] 🎉 We release the dataflow-agent.
- [2025-06-30] 🎉 We release the documentation of dataflow.
<!-- - [2025-05-30] 🎉 We added two data processing pipelines, i.e. knowledge base cleaning, and agentic rag data construction pipeline. -->
<!-- - [2025-04-30] 🎉 We added four data processing pipelines, i.e. text, code, nl2sql, and reasoning data pipeline. -->
<!-- - [2024-12-26] 🎉 Our first data evaluation and processing system is now open source. -->
- [2024-10-14] 🎉 We summarize data evaluation papers and codes in [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
- [2024-10-14] 🎉 Our first data-centric evaluation system is now open source.

## Overview
DataFlow is a data evaluation and processing system designed to **extract, clean, and augment** high-quality training data from noisy sources (PDF, plain-text, low-quality QA), thereby improving the performance of large language models in specific domains through targeted training (Pre-training, Supervised Fine-tuing, RL training). **DataFlow has been empirically validated to improve model performance in fields such as healthcare, finance, and law.**

<!-- 1. Evaluate data quality from multiple dimensions; 
2. Filter out high-quality data;
3. Generate chain-of-thought or other types of augmentation. We mainly support SOTA algorithms within academic papers with strong theoretical support. -->

<!-- We now support text, image, video, and multimodality data types. -->
Specifically, we constructing diverse `operators` leveraging rule-based methods, deep learning models, large language models (LLMs), and LLM APIs. These operators are systematically integrated into six distinct `pipelines`, collectively forming the comprehensive `Dataflow` system. Additionally, we develop an intelligent `agent` capable of dynamically assembling new `pipelines` by recombining existing `operators` on demand.


<!-- Text: 输入是烂数据 通过大模型 输出QA （主要是强化学习）
NL2SQL: 反向构造SQL QA
Reasonning：Question很短，构建长链COT ，是否有category，是否有难度（通过大模型）
Agentic RAG: 输入QA，出来是 QA。没有额外信息解决不了，必须要引入
Knowlege Base Cleaning: PDF，表格+doc text输入，输出是高质量知识库
Dataflow-agent: 用Agent自动合成pipeline。编排已有算子。 -->

## Pipelines & Agent
Current Pipelines in Dataflow are as follows:
- **Text Pipeline**: Mine question-answer pairs from large-scale plain-text data for use in SFT and RL training.
- **Reasoning Pipeline**: Enhances existing question–answer pairs with (1) extended chain-of-thought, (2) category classification, and (3) difficulty estimation.
- **Text2SQL Pipeline**: Translates natural language questions into SQL queries, supplemented with explanations, chain-of-thought reasoning, and contextual schema information.
- **Agentic RAG Pipeline**: Identify and extract QA pairs from existing QA datasets or knowledge bases that require external knowledge to answer, for use in downstream training of Agnetic RAG tasks.
- **Knowlege Base Cleaning Pipeline**: Extract and structure knowledge from unorganized sources like tables, PDFs, and Word documents into usable entries for downstream RAG or QA pair generation.


Building on top of this, we also provide the **DataFlow Agent**, which can arrange existing `operators` and automatically construct new pipelines based on task requirements.


## Quick Start
For environment setup and installation, please using the following commands👇

```shell
conda create -n dataflow python=3.10
conda activate dataflow

git clone https://github.com/Open-DataFlow/DataFlow
cd DataFlow
pip install -e .
```

For **Quick-Start** and **Guide**, please visit or [Documentation](https://open-dataflow.github.io/DataFlow-Doc/).


## Features & Visualization

### 1. Text PipeLine


### 2. Reasoning Pipeline
![](./static/images/demo_reasoning.png)

For demo inputs and outputs, you can refence our [Reasoning Pipeline sample](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning/) on Huggingface.

- Performance Boost：
  - ![](./static/images/reasoning_performance.png)



### 3. Text2SQL PipeLine


## Citation
```
@article{wang2025rare,
  title={Rare: Retrieval-augmented reasoning modeling},
  author={Wang, Zhengren and Yu, Jiayang and Ma, Dongsheng and Chen, Zhe and Wang, Yu and Li, Zhiyu and Xiong, Feiyu and Wang, Yanfeng and Tang, Linpeng and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2503.23513},
  year={2025}
}
```

## Statistics
<a href="https://star-history.com/#Open-DataFlow/DataFlow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
 </picture>
</a>

