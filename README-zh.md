# DataFlow

<div align="center">
  <img src="https://github.com/user-attachments/assets/3fe636ad-3026-4faf-aa44-c84b8f97a05d">
<!-- [![](https://img.shields.io/github/forks/OpenDCAI/DataFlow?style=social)](https://github.com/OpenDCAI/DataFlow) -->

[![](https://img.shields.io/github/stars/OpenDCAI/DataFlow?style=social)](https://github.com/OpenDCAI/DataFlow)
[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/opendcai/DataFlow)](https://github.com/OpenDCAI/DataFlow/issues?q=is%3Aissue%20state%3Aclosed)
[![](https://img.shields.io/github/issues-pr-raw/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/pulls)
[![issue resolution](https://img.shields.io/github/issues-pr-closed-raw/opendcai/DataFlow)](https://github.com/OpenDCAI/DataFlow/pulls?q=is%3Apr+is%3Aclosed)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlow?color=green)](https://github.com/OpenDCAI/DataFlow)

[![PyPI version](https://img.shields.io/pypi/v/open-dataflow)](https://pypi.org/project/open-dataflow/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/open-dataflow)](https://pypi.org/project/open-dataflow/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/open-dataflow?style=flat&logo=python)](https://pypistats.org/packages/open-dataflow)
[![Downloads](https://static.pepy.tech/badge/open-dataflow)](https://pepy.tech/project/open-dataflow)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1haosl2QS4N4HM7u7HvSsz_MnLabxexXl?usp=sharing)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/molyheci/dataflow)
[![Documents](https://img.shields.io/badge/官方文档-单击此处-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlow-Doc/)
[![Arxiv](https://img.shields.io/badge/技术报告-2512.16676-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2512.16676)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/DataFlow)

[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://github.com/user-attachments/assets/3c2e5d4d-d1ea-4d8c-9146-ff14e657e857)

<a href="https://trendshift.io/repositories/16045" target="_blank"><img src="https://trendshift.io/api/badge/repositories/16045" alt="OpenDCAI%2FDataFlow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dd/open-dataflow?style=flat&logo=python) -->
<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dw/open-dataflow?style=flat&logo=python) -->
<!-- [![](https://img.shields.io/github/license/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/blob/main/LICENSE) -->
<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/commits/main/) -->

🎉 如果你认可我们的项目，欢迎在 GitHub 上点个 ⭐ Star，关注项目最新进展。

**新手友好的学习资源（持续更新）**:  
[[🎬 视频教程]](https://space.bilibili.com/3546929239689711?spm_id_from=333.337.0.0)  
[[📚 图文教程]](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)

简体中文 | [English](./README.md)

</div>

## 📰 1. 最新动态
- **[2025-12-19] 🎉 我们的 DataFlow 技术报告正式发布！**  
  如果我们的工作对你有所帮助，欢迎查阅并引用。  
  👉 在 arXiv 阅读完整报告：[https://arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676)

- **[2025-11-20] DataFlow 全新上线 Data Agents！** 🤖  欢迎体验，并可查看 [Bilibili](https://space.bilibili.com/3546929239689711/lists/6761342?type=season) 上的教程快速上手。

- [2025-06-28] 🎉 我们全新发布的以数据为中心的系统**DataFlow**已开源 —— 敬请关注后续更新！

## 🔍 2. 项目概述

<!-- ![dataflow_framework](https://github.com/user-attachments/assets/b44db630-754a-44a8-bec7-6d350bf5ed61) -->
  
![df_overview_final_300](https://github.com/user-attachments/assets/57dd0838-6e24-4814-a89a-02ca0667bd5c)

DataFlow 是一个数据准备系统，旨在从噪声数据源（PDF、纯文本、低质量问答）中**解析，生成，加工并评估高质量数据**，以提升大语言模型（LLMs）在特定领域的表现，支持预训练、监督微调（SFT）、强化学习训练以及基于知识库的 RAG 系统。**我们在医疗、金融和法律等多个垂类领域实证验证了 DataFlow 的有效性。**

我们构建了多种基于规则、深度学习、大语言模型及其 API 的 `数据算子（Operators）`，并将其系统性地整合为多条 `数据流水线（Pipelines）`，共同组成完整的 `DataFlow 系统`。此外，我们还构建了智能的 `DataFlow-Agent`，支持按需动态编排已有算子，合成新的数据流水线。

## 🛠️ 3. 数据算子功能介绍

### 🔧 3.1 算子工作机制

DataFlow采用模块化的算子设计理念，通过组合不同类型的算子来构建灵活的数据处理流水线。算子作为数据处理的基本单元，能够接收结构化数据输入（如json/jsonl/csv格式），经过智能处理后输出高质量的数据结果。详细的算子使用指南请参考：[项目文档](https://opendcai.github.io/DataFlow-Doc/zh/guide/text_evaluation_operators/)

![dataflow_operator](https://github.com/user-attachments/assets/d79a0d8b-09ef-457e-af8b-85af0d03b73d)

### 📊 3.2 算子分类体系

在DataFlow框架中，算子按功能特性分为三大核心类别：

| 算子类型 | 数量 | 主要功能 |
|---------|------|----------|
| **通用算子 (Generic Operators)** | 80+ | 涵盖文本评估、处理和合成的通用功能 |
| **领域专用算子 (Domain-Specific Operators)** | 40+ | 针对特定领域（如医疗、金融、法律）的专业处理 |
| **评估算子 (Evaluation Operators)** | 20+ | 从6个维度全面评估数据质量 |

## 🛠️ 4. 数据流程功能介绍

### 🔧 4.1 推荐使用的完整流水线

目前 DataFlow 包含以下主要数据处理流程：

- [📝 **文本处理流程（Text Pipeline）**](https://opendcai.github.io/DataFlow-Doc/zh/guide/textpipeline)：从大规模纯文本（多为网络爬取）中挖掘问答对，用于监督微调和强化学习训练。
  - ![dataflow_text_pipeline](https://github.com/user-attachments/assets/34e3aef2-ba4f-4997-9127-9d21fdb2dede)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text)

- [🧠 **推理流程（Reasoning Pipeline）**](https://opendcai.github.io/DataFlow-Doc/zh/guide/reasoningpipeline/#_2-question-handling)：增强已有问答对，添加 (1) 长链式推理（Chain-of-Thought），(2) 类别标注，(3) 难度估计。
  - ![dataflow_reasoning_pipeline](https://github.com/user-attachments/assets/fef5829b-3991-4dcb-99ad-d61d95c982ea)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning)

- [🗃️ **Text2SQL 流程**](https://opendcai.github.io/DataFlow-Doc/zh/guide/text2sqlpipeline/)：将自然语言问题转化为 SQL 查询，辅以解释、思维链推理和数据库结构上下文信息。
  - ![dataflow_text2sql_pipeline](https://github.com/user-attachments/assets/bae9914e-851b-4502-8696-291d6c1b8824)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text2SQL)

- [📚 **知识库清洗流程**](https://opendcai.github.io/DataFlow-Doc/zh/guide/r51ooua8/)：从表格、PDF 和 Word 文档等非结构化数据源中提取并整理知识，将其转化为可用于下游 RAG 或 QA 配对生成的可用条目。
  - ![dataflow_KnowledgeBaseClean_pipeline](https://github.com/user-attachments/assets/6f21e682-ec10-42af-b5e2-8fec2929eeae)

- [🤖 **Agent式RAG流程**](https://opendcai.github.io/DataFlow-Doc/zh/guide/agenticrag_pipeline/)：从已有问答或知识库中挖掘需要外部知识才能作答的问答对，用于训练 Agentic RAG 模型。
  - ![dataflow_agenticRAG_pipeline](https://github.com/user-attachments/assets/65e80dca-f286-495b-abb7-804b3fc34a53)


### ⚙️ 4.2 算子自由组合的灵活流水线

在本框架中，算子可灵活组合构建数据处理流水线，按功能分为基础算子（Fundamental Operators）、通用算子（Generic Operators）、领域特定算子（Domain-Specific Operators）和评估算子（Evaluation Operators）等，覆盖从清洗到评估的多种任务。详见[项目文档](https://OpenDCAI.github.io/DataFlow-Doc/)了解具体用法。

### 🤖 4.3 Agent驱动的流水线自动编排

- **DataFlow-Agent**：智能助手，可执行数据分析、编写自定义算子，并根据任务目标自动编排算子构建数据处理流水线。
  - ![dataflow_agent_pipeline](https://github.com/user-attachments/assets/fe0776fa-55bd-49cd-bfe6-06ad377f62bb)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Agent)

## ⚡ 5. 快速开始

### 🛠️ 5.1 环境配置和安装

请使用如下命令进行环境配置和安装👇

```shell
conda create -n dataflow python=3.10 
conda activate dataflow

pip install open-dataflow
```
如果你想要用你自己的GPU完成本地推理，则需要:
```shell
pip install open-dataflow[vllm]
```
> Dataflow 支持 Python>=3.10 的环境

安装完成后，你可以用如下指令查看dataflow是否正确安装:
```shell
dataflow -v
```

如果安装正确，应该会看到:
```log
open-dataflow codebase version: 1.0.0
        Checking for updates...
        Local version:  1.0.0
        PyPI newest version:  1.0.0
You are using the latest version: 1.0.0.
```

#### 🐳 5.1.1 Docker安装（可选方式）

我们还提供了 **Dockerfile** 以便于部署，同时也提供了**预构建的 Docker 镜像**供您直接使用。

##### 方式一：使用预构建的 Docker 镜像

您可以直接拉取并使用我们预构建的 Docker 镜像：

```shell
# 拉取预构建镜像
docker pull molyheci/dataflow:cu124

# 使用 GPU 支持运行容器
docker run --gpus all -it molyheci/dataflow:cu124

# 在容器内验证安装
dataflow -v
```

##### 方式二：从 Dockerfile 构建

或者，您也可以从项目提供的 Dockerfile 构建镜像：

```shell
# 克隆代码仓库（HTTPS 方式）
git clone https://github.com/OpenDCAI/DataFlow.git
# 或使用 SSH 方式
# git clone git@github.com:OpenDCAI/DataFlow.git

cd DataFlow

# 构建 Docker 镜像
docker build -t dataflow:custom .

# 运行容器
docker run --gpus all -it dataflow:custom

# 在容器内验证安装
dataflow -v
```

> **注意**：Docker 镜像包含 CUDA 12.4.1 支持，并预装了 vLLM 用于 GPU 加速。请确保您已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 以使用 GPU 功能。

### 🚀 5.2 用Google Colab快速开始
你可以直接在 Google Colab 上启动你的第一个 DataFlow 翻译项目，无需任何本地环境配置。
按照 Notebook 中提供的指导流程，你可以从基础的翻译示例逐步扩展到更复杂的 DataFlow 数据处理流水线。

👉 [使用 Google Colab 启动 DataFlow](https://colab.research.google.com/drive/1haosl2QS4N4HM7u7HvSsz_MnLabxexXl?usp=sharing)

### 📖 5.3 参考DataFlow项目文档

详细**使用说明**和**入门指南**，请参考我们的 [项目文档](https://OpenDCAI.github.io/DataFlow-Doc/)。

## 🧪 6. Experimental Results
For Detailed Experiments setting, please visit our [DataFlow Technical Report](https://arxiv.org/abs/2512.16676).


### 6.1 Text Pipeline

#### 6.1.1 预训练数据过滤
我们从 SlimPajama-627B 语料库中抽取了一个 100B token 的子集，并对其应用了多种 DataFlow 文本预训练过滤器。随后，我们基于 Megatron-DeepSpeed 训练框架，从零开始训练了一个 Qwen2.5-0.5B 模型，总训练规模为 30B tokens。实验结果如下所示。

| Methods            | ARC-C | ARC-E | MMLU | HellaSwag | WinoGrande | Gaokao-MathQA | Avg   |
|--------------------|:-----:|:-----:|:----:|:---------:|:----------:|:-------------:|:-----:|
| **Random-30B**     | 25.26 | 43.94 | 27.03 | 37.02 | 50.99 | 27.35 | 35.26 |
| **Qurating-30B**   | 25.00 | 43.14 | 27.50 | 37.03 | 50.67 | 26.78 | 35.02 |
| **FineWeb-Edu-30B**| 26.45 | 45.41 | 27.41 | 38.06 | 50.43 | 25.64 | 35.57 |
| **DataFlow-30B**   | 25.51 | 45.58 | 27.42 | 37.58 | 50.67 | 27.35 | **35.69** |

#### 6.1.2 小规模 SFT 数据过滤与合成
为研究 小规模 SFT 数据的质量影响，我们使用 LLaMA-Factory 对 Qwen2.5-7B Base 模型进行了微调，所用数据集包括 WizardLM 和 Alpaca。
对于每个数据集，我们比较了 随机采样的 5K 样本 与 经过 DataFlow SFT 流水线过滤的 5K 样本 在下游性能上的差异。此外，我们还基于 DataFlow 的 Condor Generator 与 Condor Refiner 流水线 合成了一个规模为 15K 的数据集（记为 DataFlow-SFT-15K），并进一步对其应用了 DataFlow 的 SFT 过滤流程（不包含 Instagram 过滤器）。评测覆盖了 数学、代码与知识 三大类的综合基准。

### Math Benchmarks
| Methods | math | gsm8k | aime24 | minerva | olympiad | Avg |
|--------|:----:|:-----:|:------:|:-------:|:--------:|:---:|
| **Alpaca (random)** | 54.9 | 77.2 | 13.3 | 14.0 | 27.0 | 37.3 |
| **Alpaca (filtered)** | 60.3 | 80.0 | 13.3 | 14.7 | 30.7 | 39.8 |
| **WizardLM (random)** | 61.1 | 84.2 | 6.7 | 18.0 | 29.3 | 39.9 |
| **WizardLM (filtered)** | 69.7 | 88.8 | 10.0 | 19.9 | 35.4 | 44.8 |
| **DataFlow-SFT-15K (random)** | 72.6 | 89.6 | 13.3 | 37.9 | 32.9 | **49.3** |
| **DataFlow-SFT-15K (filtered)** | 73.3 | 90.2 | 13.3 | 36.0 | 35.9 | **49.7** |

---

### Code Benchmarks
| Methods | HumanEval | MBPP | Avg |
|--------|:---------:|:----:|:---:|
| **Alpaca (random)** | 71.3 | 75.9 | 73.6 |
| **Alpaca (filtered)** | 73.8 | 75.7 | 74.8 |
| **WizardLM (random)** | 75.6 | 82.0 | **78.8** |
| **WizardLM (filtered)** | 77.4 | 80.4 | **78.9** |
| **DataFlow-SFT-15K (random)** | 79.9 | 75.9 | 77.9 |
| **DataFlow-SFT-15K (filtered)** | 82.9 | 74.9 | **78.9** |

---

### Knowledge Benchmarks
| Methods | MMLU | C-EVAL | Avg |
|--------|:----:|:------:|:---:|
| **Alpaca (random)** | 71.8 | 80.0 | 75.9 |
| **Alpaca (filtered)** | 71.8 | 80.0 | 75.9 |
| **WizardLM (random)** | 71.8 | 79.2 | 75.5 |
| **WizardLM (filtered)** | 71.9 | 79.6 | 75.8 |
| **DataFlow-SFT-15K (random)** | 72.1 | 80.0 | **76.1** |
| **DataFlow-SFT-15K (filtered)** | 72.2 | 80.4 | **76.3** |

#### 6.1.3 对话数据合成
我们使用 DataFlow 的对话生成流水线 合成了 DataFlow-Chat-15K 数据集，并基于该数据对 Qwen2.5-7B-Base 模型进行了微调。对比方法包括 ShareGPT-15K、UltraChat-15K 及其 完整（未截断）版本。评测涵盖了 对话领域基准（TopDial、Light）以及 通用能力基准（MMLU、AlpacaEval、Arena-Hard）。

### Conversation Benchmarks
| Model | TopDial | Light | Avg |
|------|:-------:|:-----:|:---:|
| **Qwen2.5-7B** | 7.71 | 7.79 | 7.75 |
| **+ ShareGPT-15K** | 7.75 | 6.72 | 7.24 |
| **+ UltraChat-15K** | 7.72 | 6.83 | 7.28 |
| **+ DataFlow-Chat-15K** | **7.98** | **8.10** | **8.04** |

---

### General Benchmarks
| Model | MMLU | AlpacaEval | Arena-Hard | Avg |
|------|:----:|:----------:|:----------:|:---:|
| **Qwen2.5-7B** | 71.45 | 7.05 | 0.60 | 26.36 |
| **+ ShareGPT-15K** | 73.09 | 3.70 | 1.30 | 26.03 |
| **+ UltraChat-15K** | 72.97 | 3.97 | 0.80 | 25.91 |
| **+ DataFlow-Chat-15K** | 73.41 | **10.11** | 1.10 | **28.21** |

### 6.2 推理数据合成流水线
我们采用 NuminaMath 数据集作为高质量的种子数据集，并比较了三种不同的训练数据来源：（1）从 Open-R1 中随机采样的 10K 子集，（2）从 Synthetic-1 中随机采样的 10K 子集，以及（3）使用 DataFlow 构建的、规模为 10K 的合成数据集 DataFlow-Reasoning-10K。

| Setting | Model | gsm8k | math | amc23 | olympiad | gaokao24_mix | minerva | AIME24@32 | AIME25@32 | Avg |
|--------|-------|:-----:|:----:|:-----:|:--------:|:-------------:|:--------:|:---------:|:---------:|:----:|
| Baseline | **Qwen2.5-32B-Instruct** | 95.8 | 73.5 | 70.0 | 38.5 | 42.9 | 26.5 | 16.8 | 11.6 | 46.95 |
| 1 Epoch | **+ SYNTHETIC-1-10k** | 92.9 | 71.8 | 52.5 | 38.4 | 23.1 | 24.3 | 35.6 | 34.0 | 46.6 |
| 1 Epoch | **+ Open-R1-10k** | 91.5 | 72.3 | 65.0 | 38.4 | 20.9 | 24.6 | 43.0 | 33.5 | 48.7 |
| 1 Epoch | **+ DataFlow-Reasoning-10K** | 93.9 | 72.3 | 72.5 | 38.7 | 38.5 | 26.5 | 35.9 | 34.5 | **51.6** |
| 2 Epochs | **+ SYNTHETIC-1-10k** | 94.5 | 78.4 | 75.0 | 45.0 | 24.2 | 28.3 | 48.4 | 37.9 | 54.0 |
| 2 Epochs | **+ Open-R1-10k** | 93.9 | 77.2 | 80.0 | 44.1 | 20.9 | 25.4 | 51.0 | 40.7 | 54.2 |
| 2 Epochs | **+ DataFlow-Reasoning-10K** | 94.4 | 76.6 | 75.0 | 45.2 | 42.9 | 25.7 | 45.4 | 40.0 | **55.7** |

### 6.3 代码数据构建流水线
我们从 Ling-Coder-SFT 语料库中随机采样 20K 条实例，并将其输入 DataFlow Code Pipeline 进行处理，从而得到三个不同规模的高质量代码指令数据集：DataFlow-Code-1K、DataFlow-Code-5K 和 DataFlow-Code-10K。这些数据集旨在为代码生成任务提供经过流水线精炼的高质量监督信号。
我们将所合成的数据集与 Code-Alpaca-1K 以及 Self-OSS-Instruct-SC2-Exec-Filter-1K 进行对比评测。

#### Trained on Qwen2.5-7B-Instruct
| Training Data | BigCodeBench | LiveCodeBench (v6) | CruxEval (Input) | CruxEval (Output) | HumanEval+ | Avg |
|--------------|:------------:|:------------------:|:----------------:|:-----------------:|:----------:|:---:|
| **Qwen2.5-7B-Instruct** | 35.3 | 23.4 | 44.8 | 43.9 | 72.6 | 44.0 |
| **+ Code Alpaca-1K** | 33.3 | 18.7 | 45.6 | 46.4 | 66.5 | 42.1 |
| **+ Self-OSS** | 31.9 | 21.4 | 46.9 | 45.9 | 70.1 | 43.2 |
| **+ DataFlow-Code-1K** | 35.5 | 25.7 | 48.0 | 45.1 | 72.6 | 45.4 |
| **+ DataFlow-Code-5K** | 36.2 | **26.4** | 48.6 | 45.0 | 73.2 | 45.9 |
| **+ DataFlow-Code-10K** | **36.8** | 26.0 | **48.8** | **45.4** | **73.8** | **46.2** |

---

#### Trained on Qwen2.5-14B-Instruct
| Training Data | BigCodeBench | LiveCodeBench (v6) | CruxEval (Input) | CruxEval (Output) | HumanEval+ | Avg |
|--------------|:------------:|:------------------:|:----------------:|:-----------------:|:----------:|:---:|
| **Qwen2.5-14B-Instruct** | 37.5 | 33.4 | 48.0 | 48.5 | 74.4 | 48.4 |
| **+ Code Alpaca-1K** | 37.0 | 28.2 | 50.2 | 49.6 | 71.3 | 47.3 |
| **+ Self-OSS** | 36.9 | 22.3 | 52.6 | 50.1 | 68.3 | 46.0 |
| **+ DataFlow-Code-1K** | 41.4 | **33.7** | 51.0 | 50.9 | **77.3** | 50.9 |
| **+ DataFlow-Code-5K** | 41.1 | 33.2 | 52.5 | 50.6 | 76.2 | 50.7 |
| **+ DataFlow-Code-10K** | **41.9** | 33.2 | **52.9** | **51.0** | 76.2 | **51.0** |


## 📄 7. 发表论文

我们团队已发表以下论文，并作为构成DataFlow系统的核心组件：

| 论文标题 | DataFlow组件 | 会议 | 年份 |
|---------|-------------|:------:|------|
| [Let's Verify Math Questions Step by Step](https://arxiv.org/abs/2505.13903) | Math question quality evaluation | KDD | 2026 |
| [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](https://arxiv.org/pdf/2502.13383) | 多模态推理验证框架，用于数据处理和评估 | ACL | 2025 |
| [Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration](https://arxiv.org/pdf/2410.08102) | 多智能体协作数据选择机制，增强数据筛选和处理能力 | ACL | 2025 |

**合作机构**: 
<img src="./static/logo/pku.png" alt="PKU" height="30"/> 
<img src="./static/logo/hkust.png" alt="HKUST" height="30"/> 
<img src="./static/logo/CAS.png" alt="CAS" height="30"/> 
<img src="./static/logo/shanghai_ailab.png" alt="Shanghai AI Lab" height="30"/> 
<img src="./static/logo/baichuan.png" alt="Baichuan" height="30"/> 
<img src="./static/logo/ant_group.png" alt="Ant Group" height="30"/>


## 🏆 8. 获奖与荣誉

我们荣获了两项国际顶级人工智能竞赛的**第一名**，展示了 DataFlow 系统在数据智能与推理任务中的卓越性能与创新性：

| 比赛名称                                                              | 赛道                | 奖项         | 主办方                                       | 时间         |
| ----------------------------------------------------------------- | ----------------- | ---------- | ----------------------------------------- | ---------- |
| **ICML 2025 自动化数学推理挑战赛（Automated Math Reasoning and Extensions）** | 赛道二：基于图表与表达式的物理推理 | 🥇 **第一名** | ICML AI for Math Workshop & AWS Codabench | 2025年7月18日 |
| **2025 智源语言与智能技术竞赛（LIC）**                                         | 赛道二：智源研究院赛道       | 🥇 **一等奖** | 智源研究院 & 百度                                | 2025年8月10日 |

<div align="center">

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/8f28e0fe-c883-42c0-b224-3693f6281a14" alt="ICML 2025 Certificate" width="95%"><br>
      <sub><em>ICML 2025 Automated Math Reasoning Challenge — First Place Winner</em></sub>
    </td>
    <td align="center" width="30%">
      <img src="https://github.com/user-attachments/assets/364618b6-4dfa-4c34-928f-e3da85cbd03a" alt="LIC 2025 Certificate" width="95%"><br>
      <sub><em>BAAI Language & Intelligence Challenge 2025 — First Prize</em></sub>
    </td>
  </tr>
</table>

</div>

## 💐 9. 致谢
我们衷心感谢 [MinerU](https://github.com/opendatalab/MinerU) 的卓越工作，其强大的 PDF/文档文本提取功能为数据加载提供了关键支持。
同时，我们感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 提供的高效、易用的大模型微调框架，为我们在模型训练与实验流程上的快速迭代带来了极大便利。  
感谢所有开源社区的贡献者，你们的工作共同推动了 DataFlow 的发展。
## 🤝 10. 社区与支持

欢迎加入 DataFlow 开源社区，提出问题、分享想法、与其他开发者一起共建项目！

•	📮 [GitHub Issues](../../issues)：提交 Bug 或功能建议。

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进。

•	💬 欢迎扫码加入下方社群（微信群、小红书、Twitter），与我们和其他开发者互动交流~

<div align="center">
  <img src="https://github.com/user-attachments/assets/090b8a20-6193-41b3-88a1-fe3f4791cb95" width="60%">
</div>


## 📜 11. 引用

如果 DataFlow 对你的研究或项目有帮助，欢迎引用支持我们：

```bibtex
@article{liang2025dataflow,
  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
  journal={arXiv preprint arXiv:2512.16676},
  year={2025}
}
```

## 📊 12. 统计信息
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
    想了解更多？欢迎关注我们
    <a href="https://zwt233.github.io/" target="_blank"><strong>PKU-DCAI 课题组</strong></a>，小红书账号：<strong>26133106768</strong>
  </sub>
</div>
