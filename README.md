# DataFlow

<div align="center">
  <img src="./static/images/Face.png">


[![](https://img.shields.io/github/license/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/Open-DataFlow/DataFlow?style=social)](https://github.com/Open-DataFlow/DataFlow)
[![](https://img.shields.io/github/issues-raw/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/issues)
[![](https://img.shields.io/github/last-commit/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/commits/main/)
[![](https://img.shields.io/github/contributors/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/graphs/contributors)

[简体中文](./README.zh-CN.md) | English


**[Features](#Features) • [Quick Start](#Quick_Start) • [Documentation](https://open-dataflow.github.io/DataFlow-Doc/) • [Experiments](#Experiments)**

</div>


## 1 News
- [2025-07-25] 🎉 We release the dataflow-agent.
- [2025-06-30] 🎉 We release the documentation of dataflow.
<!-- - [2025-05-30] 🎉 We added two data processing pipelines, i.e. knowledge base cleaning, and agentic rag data construction pipeline. -->
<!-- - [2025-04-30] 🎉 We added four data processing pipelines, i.e. text, code, nl2sql, and reasoning data pipeline. -->
<!-- - [2024-12-26] 🎉 Our first data evaluation and processing system is now open source. -->
- [2024-10-14] 🎉 We summarize data evaluation papers and codes in [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
- [2024-10-14] 🎉 Our first data-centric evaluation system is now open source.

## 2 Overview

  <img src="./static/images/dataflow_framework.jpg">

DataFlow is a data evaluation and processing system designed to **clean, augment and evaluate** high-quality data from noisy sources (PDF, plain-text, low-quality QA), thereby improving the performance of large language models (LLMs) in specific domains through targeted training (Pre-training, Supervised Fine-tuing, RL training) or RAG using knowledge base cleaning. **DataFlow has been empirically validated to improve domain-oriented LLM's performance in fields such as healthcare, finance, and law.**

Specifically, we constructing diverse `operators` leveraging rule-based methods, deep learning models, LLMs, and LLM APIs. These operators are systematically integrated into distinct `pipelines`, collectively forming the comprehensive `Dataflow system`. Additionally, we develop an intelligent `Dataflow-agent` capable of dynamically assembling new `pipelines` by recombining existing `operators` on demand.



<!-- Text: 输入是烂数据 通过大模型 输出QA （主要是强化学习）
NL2SQL: 反向构造SQL QA
Reasonning：Question很短，构建长链COT ，是否有category，是否有难度（通过大模型）
Agentic RAG: 输入QA，出来是 QA。没有额外信息解决不了，必须要引入
Knowlege Base Cleaning: PDF，表格+doc text输入，输出是高质量知识库
Dataflow-agent: 用Agent自动合成pipeline。编排已有算子。 -->

## 3 Pipelines Functionality
Current Pipelines in Dataflow are as follows:
- **Text Pipeline**: Mine question-answer pairs from large-scale plain-text data (mostly crawed from InterNet) for use in SFT and RL training.
  - ![](./static/images/dataflow_text_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Text Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text)
- **Reasoning Pipeline**: Enhances existing question–answer pairs with (1) extended chain-of-thought, (2) category classification, and (3) difficulty estimation.
  - ![](./static/images/dataflow_reasoning_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Reasoning Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning)
- **Text2SQL Pipeline**: Translates natural language questions into SQL queries, supplemented with explanations, chain-of-thought reasoning, and contextual schema information.
  - ![](./static/images/dataflow_text2sql_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Text2SQL Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text2SQL)
- **Agentic RAG Pipeline**: Identify and extract QA pairs from existing QA datasets or knowledge bases that require external knowledge to answer, for use in downstream training of Agnetic RAG tasks.

<!-- - **Knowlege Base Cleaning Pipeline**: Extract and structure knowledge from unorganized sources like tables, PDFs, and Word documents into usable entries for downstream RAG or QA pair generation. -->



Building on top of this, we also provide the
- **DataFlow Agent**: Can arrange existing `operators` and automatically construct new pipelines based on task requirements.
  - [[HuggingFace🤗 demo input & output for **DataFlow Agent**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Agent)


<!-- ### 3.1 Text Pipeline
![](./static/images/demo_reasoning.png) -->

## 4 Quick Start
For environment setup and installation, please using the following commands👇

```shell
conda create -n dataflow python=3.10
conda activate dataflow

git clone https://github.com/Open-DataFlow/DataFlow
cd DataFlow
pip install -e .
```

For **Quick-Start** and **Guide**, please visit or [Documentation](https://open-dataflow.github.io/DataFlow-Doc/).


## 5 Experimental Results
For Detailed Experiments setting, please visit 


### 5.1 Text PipeLine

#### 5.1.1 Pre-training data filter pipeline
The `pre-training data processing pipeline` was applied to randomly sampled data from the RedPajama dataset, resulting in a final data retention rate of 13.65%. The analysis results using `QuratingScorer` are shown in the figure. As can be seen, the filtered pretraining data significantly outperforms the original data across four scoring dimensions: writing style, requirement for expert knowledge, factual content, and educational value. This demonstrates the effectiveness of the DataFlow pretraining data processing.

![alt text](./static/images/text-pretrain.png)

#### 5.1.2 SFT data filter pipeline
We filted 3k record from `alpaca` dataset and compare it with radom selected 3k data from `alpaca` dataset by training it on Qwen2.5-7B. Results are:


![](./static/images/text-sft.png)
### 2. Reasoning Pipeline

We verify our reasoning pipeline by SFT on a Qwen2.5-32B-Instruct with Reasoning Pipeline synsthized data. We generated 1k and 5k SFT data pairs. Results are: 

- ![](./static/images/reasoning_performance.png)



### 3. Text2SQL PipeLine
We fine-tuned the Qwen2.5-Coder-7B model on the Bird dataset using both Supervised Fine-tuning (SFT) and Reinforcement Learning (RL), with data constructed via the DataFlow-Text2SQL Pipeline. Results are:
![alt text](./static/images/text2sql.png)

### 
## Citation
```
@article{wang2025rare,
  title={Rare: Retrieval-augmented reasoning modeling},
  author={Wang, Zhengren and Yu, Jiayang and Ma, Dongsheng and Chen, Zhe and Wang, Yu and Li, Zhiyu and Xiong, Feiyu and Wang, Yanfeng and Tang, Linpeng and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2503.23513},
  year={2025}
}
```

<!-- ## Statistics
<a href="https://star-history.com/#Open-DataFlow/DataFlow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Open-DataFlow/DataFlow&type=Date" />
 </picture>
</a>
 -->
