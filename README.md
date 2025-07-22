# DataFlow

<div align="center">
  <img src="./static/images/Face.jpg">


[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlow-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/DataFlow?style=social)](https://github.com/OpenDCAI/DataFlow)
[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/issues)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlow?color=green)](https://github.com/OpenDCAI/DataFlow)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/commits/main/) -->

🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.

[简体中文](./README-zh.md) | English


**[🚀 Features](#-3-pipelines-functionality) • [⚡ Quick Start](#-4-quick-start) • [📖 Documentation](https://OpenDCAI.github.io/DataFlow-Doc/) • [🧪 Experiments](#-5-experimental-results)**

</div>

https://github.com/user-attachments/assets/05e047a5-99bb-4043-bc71-2b5ccdab2126

## 📰 1. News
🎉 [2025-06-28] We’re excited to announce that DataFlow, our Data-centric AI system, is now released! Stay tuned for future updates.

## 🔍 2. Overview

  <img src="./static/images/dataflow_framework.jpg">

DataFlow is a data preparation and training system designed to **parse, generate, process and evaluate** high-quality data from noisy sources (PDF, plain-text, low-quality QA), thereby improving the performance of large language models (LLMs) in specific domains through targeted training (Pre-training, Supervised Fine-tuing, RL training) or RAG using knowledge base cleaning. **DataFlow has been empirically validated to improve domain-oriented LLM's performance in fields such as healthcare, finance, and law.**

Specifically, we constructing diverse `operators` leveraging rule-based methods, deep learning models, LLMs, and LLM APIs. These operators are systematically integrated into distinct `pipelines`, collectively forming the comprehensive `DataFlow system`. Additionally, we develop an intelligent `DataFlow-agent` capable of dynamically assembling new `pipelines` by recombining existing `operators` on demand.



<!-- Text: 输入是烂数据 通过大模型 输出QA （主要是强化学习）
NL2SQL: 反向构造SQL QA
Reasonning：Question很短，构建长链COT ，是否有category，是否有难度（通过大模型）
Agentic RAG: 输入QA，出来是 QA。没有额外信息解决不了，必须要引入
Knowlege Base Cleaning: PDF，表格+doc text输入，输出是高质量知识库
Dataflow-agent: 用Agent自动合成pipeline。编排已有算子。 -->

## 🛠️ 3. Pipelines Functionality
### 🔧 3.1 Ready-to-Use PipeLines
Current Pipelines in Dataflow are as follows:
- 📝 **Text Pipeline**: Mine question-answer pairs from large-scale plain-text data (mostly crawed from InterNet) for use in SFT and RL training.
  - ![](./static/images/dataflow_text_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Text Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text)
- 🧠 **Reasoning Pipeline**: Enhances existing question–answer pairs with (1) extended chain-of-thought, (2) category classification, and (3) difficulty estimation.
  - ![](./static/images/dataflow_reasoning_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Reasoning Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning)
- 🗃️ **Text2SQL Pipeline**: Translates natural language questions into SQL queries, supplemented with explanations, chain-of-thought reasoning, and contextual schema information.
  - ![](./static/images/dataflow_text2sql_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **Text2SQL Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text2SQL)
- 📚 **Knowlege Base Cleaning Pipeline**: Extract and structure knowledge from unorganized sources like tables, PDFs, and Word documents into usable entries for downstream RAG or QA pair generation.
  - ![](./static/images/dataflow_KnowledgeBaseClean_pipeline.jpg)
- 🤖 **Agentic RAG Pipeline**: Identify and extract QA pairs from existing QA datasets or knowledge bases that require external knowledge to answer, for use in downstream training of Agnetic RAG tasks.
  - ![](./static/images/dataflow_agenticRAG_pipeline.jpg)
### ⚙️ 3.2 Flexible Operator PipeLines
In this framework, operators are categorized into Fundamental Operators, Generic Operators, Domain-Specific Operators, and Evaluation Operators, etc., supporting data processing and evaluation functionalities. Please refer to the [documentation](https://OpenDCAI.github.io/DataFlow-Doc/) for details.

### 🤖 3.3 Agent Guided Pipelines
<!-- Building on top of this, we also provide the -->
- **DataFlow Agent**: An intelligent assistant that performs data analysis, writes custom `operators`, and automatically orchestrates them into `pipelines` based on specific task objectives.

  - ![](./static/images/dataflow_agent_pipeline.jpg)
  - [[HuggingFace🤗 demo input & output for **DataFlow Agent**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Agent)

<!-- ### 3.1 Text Pipeline
![](./static/images/demo_reasoning.png) -->

## ⚡ 4. Quick Start
For environment setup and installation, please using the following commands👇

```shell
conda create -n dataflow python=3.10 
conda activate dataflow

pip install open-dataflow
```
If you want to use your own GPU to inference locally, please use:
```shell
pip install open-dataflow[vllm]
```
> Dataflow supports Python>=3.10

You can use follwing command to check if installed correctly:
```shell
dataflow -v
```

You are expected to see following outputs:
```log
open-dataflow codebase version: 1.0.0
        Checking for updates...
        Local version:  1.0.0
        PyPI newest version:  1.0.0
You are using the latest version: 1.0.0.
```

### Using Gradio to Test DataFlow Operators

You can quickly launch a Gradio-based interface to test DataFlow operators with the following command:

```bash
dataflow webui
```

This will start an interactive web UI, allowing you to visualize all operators seamlessly.

For **Quick-Start** and **Guide**, please visit our [Documentation](https://OpenDCAI.github.io/DataFlow-Doc/). 

[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlow-Doc/)


## 🧪 5. Experimental Results
For Detailed Experiments setting, please visit our documentation.


### 📝 5.1 Text PipeLine

#### 5.1.1 Pre-training data filter pipeline
The `pre-training data processing pipeline` was applied to randomly sampled data from the RedPajama dataset, resulting in a final data retention rate of 13.65%. The analysis results using `QuratingScorer` are shown in the figure. As can be seen, the filtered pretraining data significantly outperforms the original data across four scoring dimensions: writing style, requirement for expert knowledge, factual content, and educational value. This demonstrates the effectiveness of the DataFlow pretraining data processing.

<div align="center">
  <img src="./static/images/text-pretrain.png" width="60%">
</div>

#### 5.1.2 SFT data filter pipeline
We filted 3k record from `alpaca` dataset and compare it with radom selected 3k data from `alpaca` dataset by training it on Qwen2.5-7B. Results are:

<div align="center">
  <img src="./static/images/text-sft.png" width="60%">
</div>

### 🧠 5.2 Reasoning Pipeline

We verify our reasoning pipeline by SFT on a Qwen2.5-32B-Instruct with Reasoning Pipeline synsthized data. We generated 1k and 5k SFT data pairs. Results are: 

<div align="center">
  <img src="./static/images/reasoning_performance.png" width="60%">
</div>

### 🗃️ 5.3 Text2SQL PipeLine
We fine-tuned the Qwen2.5-Coder-7B-Instruct model using both Supervised Fine-tuning (SFT) and Reinforcement Learning (RL), with data constructed via the DataFlow-Text2SQL Pipeline. Results are:

<div align="center">
  <img src="./static/images/text2sql.png" width="60%">
</div>

## 📄 6. Publications
Our team has published the following papers that form core components of the DataFlow system:

| Paper Title | DataFlow Component | Venue | Year |
|-------------|-------------------|-------|------|
| [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](https://arxiv.org/pdf/2502.13383) | Multimodal reasoning verification framework for data processing and evaluation | ACL | 2025 |
| [Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration](https://arxiv.org/pdf/2410.08102) | Multi-actor collaborative data selection mechanism for enhanced data filtering and processing | ACL | 2025 |

**Contributing Institutions**: 
<img src="./static/logo/pku.png" alt="PKU" height="30"/> 
<img src="./static/logo/hkust.png" alt="HKUST" height="30"/> 
<img src="./static/logo/CAS.png" alt="CAS" height="30"/> 
<img src="./static/logo/shanghai_ailab.png" alt="Shanghai AI Lab" height="30"/> 
<img src="./static/logo/baichuan.png" alt="Baichuan" height="30"/> 
<img src="./static/logo/ant_group.png" alt="Ant Group" height="30"/>

## 💐 7. Acknowledgements
We sincerely appreciate [MinerU](https://github.com/opendatalab/MinerU)'s outstanding contribution, particularly its robust text extraction capabilities from PDFs and documents, which greatly facilitates data loading.

## 🤝 8. Community & Support
Join the DataFlow open-source community to ask questions, share ideas, and collaborate with other developers!

•	📮 [GitHub Issues](../../issues): Report bugs or suggest features
 
•	🔧 [GitHub Pull Requests](../../pulls): Contribute code improvements

•	💬 Join our community groups to connect with us and other contributors!
 
<div align="center">
  <img src="./static/images/community_en.jpg" width="60%">
</div>

## 📜 9. Citation
If you use DataFlow in your research, feel free to give us a cite.
```bibtex
@misc{dataflow2025,
  author       = {DataFlow Develop Team},
  title        = {DataFlow: A Unified Framework for Data-Centric AI},
  year         = {2025},
  howpublished = {\url{https://github.com/OpenDCAI/DataFlow}},
  note         = {Accessed: 2025-07-08}
}
```

## 📊 10. Statistics
<div align="center">
  <a href="https://star-history.com/#OpenDCAI/DataFlow&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenDCAI/DataFlow&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenDCAI/DataFlow&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenDCAI/DataFlow&type=Date" style="width:50%;" />
    </picture>
  </a>
  
</div>

---
<div align="center">
  <sub>
    Connect with the 
    <a href="https://zwt233.github.io/" target="_blank"><strong>PKU-DCAI Research Team</strong></a> 
    on Xiaohongshu: <strong>26133106768</strong>
  </sub>
</div>
