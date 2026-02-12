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
[![Documents](https://img.shields.io/badge/Documentation-Click_here-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlow-Doc/)
[![Arxiv](https://img.shields.io/badge/Technical_Report-2512.16676-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2512.16676)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/DataFlow)

[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://github.com/user-attachments/assets/3c2e5d4d-d1ea-4d8c-9146-ff14e657e857)

<a href="https://trendshift.io/repositories/16045" target="_blank"><img src="https://trendshift.io/api/badge/repositories/16045" alt="OpenDCAI%2FDataFlow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- ![PyPI - Downloads](https://img.shields.io/pypi/dd/open-dataflow?style=flat&logo=python)
![PyPI - Downloads](https://img.shields.io/pypi/dw/open-dataflow?style=flat&logo=python) -->

<!-- [![](https://img.shields.io/github/license/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/blob/main/LICENSE) -->

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/commits/main/) -->

<!--[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/issues) -->

🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.

**Beginner-friendly learning resources (continuously updated)**:
 [[🎬 Video Tutorials]](https://space.bilibili.com/3546929239689711?spm_id_from=333.337.0.0)
 [[📚 Written Tutorials]](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)

[简体中文](./README-zh.md) | English

</div>

## 📰 1. News

* **[2026-02-02] 🖥️ DataFlow WebUI is now available!**
  Launch the visual pipeline builder with a single command: `dataflow webui`. Build and run DataFlow pipelines through an intuitive web interface. 👉 [WebUI Docs](#54-webui)
  <div style="display: flex; gap: 12px;">
    <img src="https://github.com/user-attachments/assets/b4f172d6-7753-4121-b981-55046a7a9e43" width="45%" />
    <img src="https://github.com/user-attachments/assets/b2147987-3b1e-4f56-9818-3d5e7440fa58" width="45%" />
  </div>
* **[2026-01-20] 🌟 Awesome Works Using DataFlow is now live!**
  A new section showcasing open-source projects and research built on DataFlow. Contributions are welcome! 👉 [Awesome Works](#awesome-dataflow)

* **[2025-12-19] 🎉 Our DataFlow technical report is now available!**
  Read and cite our work on arXiv: [https://arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676)

* **[2025-11-20] 🤖 Introducing New Data Agents for DataFlow!**
  Try them out and follow the tutorial on Bilibili: [https://space.bilibili.com/3546929239689711/lists/6761342?type=season](https://space.bilibili.com/3546929239689711/lists/6761342?type=season)

* **[2025-06-28] 🎉 DataFlow is officially released!**
  Our data-centric AI system is now public. Stay tuned for future updates.


## 🔍 2. Overview

<!--  <img src="./static/images/dataflow_framework.jpg"> -->

<!--  ![dataflow_framework](https://github.com/user-attachments/assets/b44db630-754a-44a8-bec7-6d350bf5ed61) -->

![df_overview_final_300](https://github.com/user-attachments/assets/57dd0838-6e24-4814-a89a-02ca0667bd5c)

DataFlow is a data preparation and training system designed to **parse, generate, process, and evaluate** high-quality data from noisy sources (PDF, plain-text, low-quality QA), thereby improving the performance of large language models (LLMs) in specific domains through targeted training (Pre-training, Supervised Fine-tuning, RL training) or RAG using knowledge base cleaning. **DataFlow has been empirically validated to improve domain-oriented LLMs' performance in fields such as healthcare, finance, and law.**

Specifically, we are constructing diverse `operators` leveraging rule-based methods, deep learning models, LLMs, and LLM APIs. These operators are systematically integrated into distinct `pipelines`, collectively forming the comprehensive `DataFlow system`. Additionally, we develop an intelligent `DataFlow-agent` capable of dynamically assembling new `pipelines` by recombining existing `operators` on demand.

<!-- Text: 输入是烂数据 通过大模型 输出QA （主要是强化学习）
NL2SQL: 反向构造SQL QA
Reasonning：Question很短，构建长链COT ，是否有category，是否有难度（通过大模型）
Agentic RAG: 输入QA，出来是 QA。没有额外信息解决不了，必须要引入
Knowlege Base Cleaning: PDF，表格+doc text输入，输出是高质量知识库
Dataflow-agent: 用Agent自动合成pipeline。编排已有算子。 -->

## 🛠️ 3. Operators Functionality

### 🔧 3.1 How Operators Work

DataFlow adopts a modular operator design philosophy, building flexible data processing pipelines by combining different types of operators. As the basic unit of data processing, an operator can receive structured data input (such as in json/jsonl/csv format) and, after intelligent processing, output high-quality data results. For a detailed guide on using operators, please refer to the [Operator Documentation](https://opendcai.github.io/DataFlow-Doc/zh/api/home/).

![dataflow_operator](https://github.com/user-attachments/assets/d79a0d8b-09ef-457e-af8b-85af0d03b73d)

### 📊 3.2 Operator Classification System

In the DataFlow framework, operators are divided into three core categories based on their functional characteristics:

| Operator Type                       | Quantity | Main Function                                                                 |
| ----------------------------------- | -------- | ----------------------------------------------------------------------------- |
| **Generic Operators**         | 80+      | Covers general functions for text evaluation, processing, and synthesis       |
| **Domain-Specific Operators** | 40+      | Specialized processing for specific domains (e.g., medical, financial, legal) |
| **Evaluation Operators**      | 20+      | Comprehensively evaluates data quality from 6 dimensions                      |

## 🛠️ 4. Pipelines Functionality

### 🔧 4.1 Ready-to-Use PipeLines

Current Pipelines in Dataflow are as follows:

- [📝 **Text Pipeline**](https://opendcai.github.io/DataFlow-Doc/en/guide/textpipeline): Mine question-answer pairs from large-scale plain-text data (mostly crawed from InterNet) for use in SFT and RL training.
  - ![dataflow_text_pipeline](https://github.com/user-attachments/assets/34e3aef2-ba4f-4997-9127-9d21fdb2dede)
  - [[HuggingFace🤗 demo input &amp; output for **Text Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text)
- [🧠 **Reasoning Pipeline**](https://opendcai.github.io/DataFlow-Doc/en/guide/reasoningpipeline/#_2-question-handling): Enhances existing question–answer pairs with (1) extended chain-of-thought, (2) category classification, and (3) difficulty estimation.
  - ![dataflow_reasoning_pipeline](https://github.com/user-attachments/assets/fef5829b-3991-4dcb-99ad-d61d95c982ea)
  - [[HuggingFace🤗 demo input &amp; output for **Reasoning Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning)
- [🗃️ **Text2SQL Pipeline**](https://opendcai.github.io/DataFlow-Doc/en/guide/text2sqlpipeline/): Translates natural language questions into SQL queries, supplemented with explanations, chain-of-thought reasoning, and contextual schema information.
  - ![dataflow_text2sql_pipeline](https://github.com/user-attachments/assets/bae9914e-851b-4502-8696-291d6c1b8824)
  - [[HuggingFace🤗 demo input &amp; output for **Text2SQL Pipeline**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text2SQL)
- [📚 **Knowlege Base Cleaning Pipeline**](https://opendcai.github.io/DataFlow-Doc/en/guide/r51ooua8/): Extract and structure knowledge from unorganized sources like tables, PDFs, and Word documents into usable entries for downstream RAG or QA pair generation.
  - ![dataflow_KnowledgeBaseClean_pipeline](https://github.com/user-attachments/assets/6f21e682-ec10-42af-b5e2-8fec2929eeae)
- [🤖 **Agentic RAG Pipeline**](https://opendcai.github.io/DataFlow-Doc/en/guide/agenticrag_pipeline/): Identify and extract QA pairs from existing QA datasets or knowledge bases that require external knowledge to answer, for use in downstream training of Agnetic RAG tasks.
  - ![dataflow_agenticRAG_pipeline](https://github.com/user-attachments/assets/65e80dca-f286-495b-abb7-804b3fc34a53)

### ⚙️ 4.2 Flexible Operator PipeLines

In this framework, operators are categorized into Fundamental Operators, Generic Operators, Domain-Specific Operators, and Evaluation Operators, etc., supporting data processing and evaluation functionalities. Please refer to the [documentation](https://OpenDCAI.github.io/DataFlow-Doc/) for details.

### 🤖 4.3 Agent Guided Pipelines

<!-- Building on top of this, we also provide the -->

- **DataFlow Agent**: An intelligent assistant that performs data analysis, writes custom `operators`, and automatically orchestrates them into `pipelines` based on specific task objectives.

  - ![dataflow_agent_pipeline](https://github.com/user-attachments/assets/fe0776fa-55bd-49cd-bfe6-06ad377f62bb)
  - [[HuggingFace🤗 demo input &amp; output for **DataFlow Agent**]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Agent)

<!-- ### 3.1 Text Pipeline
![](./static/images/demo_reasoning.png) -->

## ⚡ 5. Quick Start

### 🛠️ 5.1 Environment Setup and Installation

Please use the following commands for environment setup and installation👇

```shell
conda create -n dataflow python=3.10 
conda activate dataflow

pip install open-dataflow
```

If you want to use your own GPU for local inference, please use:

```shell
pip install open-dataflow[vllm]
```

> DataFlow supports Python>=3.10 environments

After installation, you can use the following command to check if dataflow has been installed correctly:

```shell
dataflow -v
```

If installed correctly, you should see:

```log
open-dataflow codebase version: 1.0.0
        Checking for updates...
        Local version:  1.0.0
        PyPI newest version:  1.0.0
You are using the latest version: 1.0.0.
```

#### 🐳 5.1.1 Docker Installation (Alternative)

We also provide a **Dockerfile** for easy deployment and a **pre-built Docker image** for immediate use.

##### Option 1: Use Pre-built Docker Image

You can directly pull and use our pre-built Docker image:

```shell
# Pull the pre-built image
docker pull molyheci/dataflow:cu124

# Run the container with GPU support
docker run --gpus all -it molyheci/dataflow:cu124

# Inside the container, verify installation
dataflow -v
```

##### Option 2: Build from Dockerfile

Alternatively, you can build the Docker image from the provided Dockerfile:

```shell
# Clone the repository (HTTPS)
git clone https://github.com/OpenDCAI/DataFlow.git
# Or use SSH
# git clone git@github.com:OpenDCAI/DataFlow.git

cd DataFlow

# Build the Docker image
docker build -t dataflow:custom .

# Run the container
docker run --gpus all -it dataflow:custom

# Inside the container, verify installation
dataflow -v
```

> **Note**: The Docker image includes CUDA 12.4.1 support and comes with vLLM pre-installed for GPU acceleration. Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed to use GPU features.

### 🚀 5.2 Quick Start with Google Colab
You can start your first DataFlow translation project directly on Google Colab.
By following the provided guidelines, you can seamlessly scale from a simple translation example to more complex DataFlow pipelines.

👉 [Start DataFlow with Google Colab](https://colab.research.google.com/drive/1haosl2QS4N4HM7u7HvSsz_MnLabxexXl?usp=sharing)

### 📖 5.3 Reference Project Documentation

For detailed **usage instructions** and **getting started guide**, please visit our [Documentation](https://OpenDCAI.github.io/DataFlow-Doc/).



<a id="54-webui"></a>

### 🖥️ 5.4 WebUI

DataFlow provides a **Web-based UI (WebUI)** for visual pipeline construction and execution.
<div style="display: flex; gap: 12px;">
  <img src="https://github.com/user-attachments/assets/b4f172d6-7753-4121-b981-55046a7a9e43" width="45%" />
  <img src="https://github.com/user-attachments/assets/b2147987-3b1e-4f56-9818-3d5e7440fa58" width="45%" />
</div>


After installing the DataFlow main repository, simply run:

```bash
dataflow webui
```

This will automatically download and launch the latest **DataFlow-WebUI** and open it in your browser
(`http://localhost:<port>/` if it does not open automatically).

#### 📚 Documentation

* Chinese: [https://wcny4qa9krto.feishu.cn/wiki/F4PDw76uDiOG42k76gGc6FaBnod](https://wcny4qa9krto.feishu.cn/wiki/F4PDw76uDiOG42k76gGc6FaBnod)
* English: [https://wcny4qa9krto.feishu.cn/wiki/SYELwZhh9ixcNwkNRnhcLGmWnEg](https://wcny4qa9krto.feishu.cn/wiki/SYELwZhh9ixcNwkNRnhcLGmWnEg)

#### 🛠️ Development Repository

* [https://github.com/OpenDCAI/DataFlow-webui](https://github.com/OpenDCAI/DataFlow-webui)


## 🧪 6. Experimental Results

For Detailed Experiments setting, please visit our [DataFlow Technical Report](https://arxiv.org/abs/2512.16676).

### 6.1 Text Pipeline

#### 6.1.1 Pre-training data filter pipeline

From the SlimPajama-627B corpus, we extract a 100B-token subset and apply multiple DataFlow text-pretraining filters. We train a Qwen2.5-0.5B
model from scratch for 30B tokens using the Megatron-DeepSpeed framework, the results are as follows:

| Methods                   | ARC-C | ARC-E | MMLU | HellaSwag | WinoGrande | Gaokao-MathQA |       Avg       |
| ------------------------- | :---: | :---: | :---: | :-------: | :--------: | :-----------: | :-------------: |
| **Random-30B**      | 25.26 | 43.94 | 27.03 |   37.02   |   50.99   |     27.35     |      35.26      |
| **Qurating-30B**    | 25.00 | 43.14 | 27.50 |   37.03   |   50.67   |     26.78     |      35.02      |
| **FineWeb-Edu-30B** | 26.45 | 45.41 | 27.41 |   38.06   |   50.43   |     25.64     |      35.57      |
| **DataFlow-30B**    | 25.51 | 45.58 | 27.42 |   37.58   |   50.67   |     27.35     | **35.69** |

#### 6.1.2 SFT data filter and synthesis pipeline

To study small-scale SFT data quality, we fine-tune the Qwen2.5-7B base model using LLaMA-Factory on WizardLM and Alpaca datasets.
For each dataset, we compared a randomly sampled set of 5K instances against a set of 5K instances filtered by DataFlow's SFT pipeline. Additionally, we synthesize a 15k-size dataset, DataFlow-SFT-15K, using DataFlow’s Condor Generator and Condor Refiner pipeline, followed by DataFlow’s SFT filtering pipeline (excluding the Instagram filter). Benchmarks include comprehensive Math, Code, and Knowledge evaluation suites.

### Math Benchmarks

| Methods                               | math | gsm8k | aime24 | minerva | olympiad |      Avg      |
| ------------------------------------- | :--: | :---: | :----: | :-----: | :------: | :------------: |
| **Alpaca (random)**             | 54.9 | 77.2 |  13.3  |  14.0  |   27.0   |      37.3      |
| **Alpaca (filtered)**           | 60.3 | 80.0 |  13.3  |  14.7  |   30.7   |      39.8      |
| **WizardLM (random)**           | 61.1 | 84.2 |  6.7  |  18.0  |   29.3   |      39.9      |
| **WizardLM (filtered)**         | 69.7 | 88.8 |  10.0  |  19.9  |   35.4   |      44.8      |
| **DataFlow-SFT-15K (random)**   | 72.6 | 89.6 |  13.3  |  37.9  |   32.9   | **49.3** |
| **DataFlow-SFT-15K (filtered)** | 73.3 | 90.2 |  13.3  |  36.0  |   35.9   | **49.7** |

---

### Code Benchmarks

| Methods                               | HumanEval | MBPP |      Avg      |
| ------------------------------------- | :-------: | :--: | :------------: |
| **Alpaca (random)**             |   71.3   | 75.9 |      73.6      |
| **Alpaca (filtered)**           |   73.8   | 75.7 |      74.8      |
| **WizardLM (random)**           |   75.6   | 82.0 | **78.8** |
| **WizardLM (filtered)**         |   77.4   | 80.4 | **78.9** |
| **DataFlow-SFT-15K (random)**   |   79.9   | 75.9 |      77.9      |
| **DataFlow-SFT-15K (filtered)** |   82.9   | 74.9 | **78.9** |

---

### Knowledge Benchmarks

| Methods                               | MMLU | C-EVAL |      Avg      |
| ------------------------------------- | :--: | :----: | :------------: |
| **Alpaca (random)**             | 71.8 |  80.0  |      75.9      |
| **Alpaca (filtered)**           | 71.8 |  80.0  |      75.9      |
| **WizardLM (random)**           | 71.8 |  79.2  |      75.5      |
| **WizardLM (filtered)**         | 71.9 |  79.6  |      75.8      |
| **DataFlow-SFT-15K (random)**   | 72.1 |  80.0  | **76.1** |
| **DataFlow-SFT-15K (filtered)** | 72.2 |  80.4  | **76.3** |

#### 6.1.3 Conversation Synthesis Pipeline

We synthesize DataFlow-Chat-15K using DataFlow's conversation-generation pipeline and fine-tune Qwen2.5-7B-Base on it. Baselines include ShareGPT-15K, UltraChat-15K, and their full (non-truncated) versions. We evaluate on domain-specific tasks (TopDial, Light) and general benchmarks (MMLU, AlpacaEval, Arena-Hard).

### Conversation Benchmarks

| Model                         |    TopDial    |     Light     |      Avg      |
| ----------------------------- | :------------: | :------------: | :------------: |
| **Qwen2.5-7B**          |      7.71      |      7.79      |      7.75      |
| **+ ShareGPT-15K**      |      7.75      |      6.72      |      7.24      |
| **+ UltraChat-15K**     |      7.72      |      6.83      |      7.28      |
| **+ DataFlow-Chat-15K** | **7.98** | **8.10** | **8.04** |

---

### General Benchmarks

| Model                         | MMLU |   AlpacaEval   | Arena-Hard |       Avg       |
| ----------------------------- | :---: | :-------------: | :--------: | :-------------: |
| **Qwen2.5-7B**          | 71.45 |      7.05      |    0.60    |      26.36      |
| **+ ShareGPT-15K**      | 73.09 |      3.70      |    1.30    |      26.03      |
| **+ UltraChat-15K**     | 72.97 |      3.97      |    0.80    |      25.91      |
| **+ DataFlow-Chat-15K** | 73.41 | **10.11** |    1.10    | **28.21** |

### 6.2 Reasoning Pipeline

We adopt the NuminaMath dataset as a high-quality seed dataset. We compare three training sources: (1) a random 10K subset from Open-R1, (2) a random 10K subset from Synthetic-1, and (3) our 10K synthesized DataFlow-Reasoning-10K dataset constructed using DataFlow.

| Setting  | Model                              | gsm8k | math | amc23 | olympiad | gaokao24_mix | minerva | AIME24@32 | AIME25@32 |      Avg      |
| -------- | ---------------------------------- | :---: | :--: | :---: | :------: | :----------: | :-----: | :-------: | :-------: | :------------: |
| Baseline | **Qwen2.5-32B-Instruct**     | 95.8 | 73.5 | 70.0 |   38.5   |     42.9     |  26.5  |   16.8   |   11.6   |     46.95     |
| 1 Epoch  | **+ SYNTHETIC-1-10k**        | 92.9 | 71.8 | 52.5 |   38.4   |     23.1     |  24.3  |   35.6   |   34.0   |      46.6      |
| 1 Epoch  | **+ Open-R1-10k**            | 91.5 | 72.3 | 65.0 |   38.4   |     20.9     |  24.6  |   43.0   |   33.5   |      48.7      |
| 1 Epoch  | **+ DataFlow-Reasoning-10K** | 93.9 | 72.3 | 72.5 |   38.7   |     38.5     |  26.5  |   35.9   |   34.5   | **51.6** |
| 2 Epochs | **+ SYNTHETIC-1-10k**        | 94.5 | 78.4 | 75.0 |   45.0   |     24.2     |  28.3  |   48.4   |   37.9   |      54.0      |
| 2 Epochs | **+ Open-R1-10k**            | 93.9 | 77.2 | 80.0 |   44.1   |     20.9     |  25.4  |   51.0   |   40.7   |      54.2      |
| 2 Epochs | **+ DataFlow-Reasoning-10K** | 94.4 | 76.6 | 75.0 |   45.2   |     42.9     |  25.7  |   45.4   |   40.0   | **55.7** |

### 6.3 Code PipeLine

We randomly sample 20k instances from the Ling-Coder-SFT corpus and process them through the DataFlow Code Pipeline. This yields three curated code instruction datasets of different scales, DataFlow-Code-1K, DataFlow-Code-5K, and DataFlow-Code-10K, each designed to provide high-quality, pipeline-refined supervision signals for code generation tasks.

We compare our synthesized datasets against Code-Alpaca-1k and Self-OSS-Instruct-SC2-Exec-Filter-1k.

#### Trained on Qwen2.5-7B-Instruct

| Training Data                 |  BigCodeBench  | LiveCodeBench (v6) | CruxEval (Input) | CruxEval (Output) |   HumanEval+   |      Avg      |
| ----------------------------- | :------------: | :----------------: | :--------------: | :---------------: | :------------: | :------------: |
| **Qwen2.5-7B-Instruct** |      35.3      |        23.4        |       44.8       |       43.9       |      72.6      |      44.0      |
| **+ Code Alpaca-1K**    |      33.3      |        18.7        |       45.6       |       46.4       |      66.5      |      42.1      |
| **+ Self-OSS**          |      31.9      |        21.4        |       46.9       |       45.9       |      70.1      |      43.2      |
| **+ DataFlow-Code-1K**  |      35.5      |        25.7        |       48.0       |       45.1       |      72.6      |      45.4      |
| **+ DataFlow-Code-5K**  |      36.2      |   **26.4**   |       48.6       |       45.0       |      73.2      |      45.9      |
| **+ DataFlow-Code-10K** | **36.8** |        26.0        |  **48.8**  |  **45.4**  | **73.8** | **46.2** |

---

#### Trained on Qwen2.5-14B-Instruct

| Training Data                  |  BigCodeBench  | LiveCodeBench (v6) | CruxEval (Input) | CruxEval (Output) |   HumanEval+   |      Avg      |
| ------------------------------ | :------------: | :----------------: | :--------------: | :---------------: | :------------: | :------------: |
| **Qwen2.5-14B-Instruct** |      37.5      |        33.4        |       48.0       |       48.5       |      74.4      |      48.4      |
| **+ Code Alpaca-1K**     |      37.0      |        28.2        |       50.2       |       49.6       |      71.3      |      47.3      |
| **+ Self-OSS**           |      36.9      |        22.3        |       52.6       |       50.1       |      68.3      |      46.0      |
| **+ DataFlow-Code-1K**   |      41.4      |   **33.7**   |       51.0       |       50.9       | **77.3** |      50.9      |
| **+ DataFlow-Code-5K**   |      41.1      |        33.2        |       52.5       |       50.6       |      76.2      |      50.7      |
| **+ DataFlow-Code-10K**  | **41.9** |        33.2        |  **52.9**  |  **51.0**  |      76.2      | **51.0** |

## 📄 7. Publications

Our team has published the following papers that form core components of the DataFlow system:

| Paper Title                                                                                                             | DataFlow Component                                                                            | Venue | Year |
| ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----- | ---- |
| [Let&#39;s Verify Math Questions Step by Step](https://arxiv.org/abs/2505.13903)                                           | Math question quality evaluation                                                              | KDD   | 2026 |
| [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](https://arxiv.org/pdf/2502.13383)           | Multimodal reasoning verification framework for data processing and evaluation                | ACL   | 2025 |
| [Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration](https://arxiv.org/pdf/2410.08102) | Multi-actor collaborative data selection mechanism for enhanced data filtering and processing | ACL   | 2025 |

**Contributing Institutions**:
<img src="./static/logo/pku.png" alt="PKU" height="30"/>
<img src="./static/logo/hkust.png" alt="HKUST" height="30"/>
<img src="./static/logo/CAS.png" alt="CAS" height="30"/>
<img src="./static/logo/shanghai_ailab.png" alt="Shanghai AI Lab" height="30"/>
<img src="./static/logo/baichuan.png" alt="Baichuan" height="30"/>
<img src="./static/logo/ant_group.png" alt="Ant Group" height="30"/>

## 🏆 8. Awards & Achievements

We are honored to have received **first-place awards** in two major international AI competitions, recognizing the excellence and robustness of DataFlow and its reasoning capabilities:

| Competition                                                               | Track                                                       | Award                          | Organizer                                                 | Date            |
| ------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------ | --------------------------------------------------------- | --------------- |
| **ICML 2025 Challenges on Automated Math Reasoning and Extensions** | Track 2:*Physics Reasoning with Diagrams and Expressions* | 🥇**First Place Winner** | ICML AI for Math Workshop & AWS Codabench                 | July 18, 2025   |
| **2025 Language and Intelligence Challenge (LIC)**                  | Track 2:*Beijing Academy of Artificial Intelligence*      | 🥇**First Prize**        | Beijing Academy of Artificial Intelligence (BAAI) & Baidu | August 10, 2025 |

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

## 💐 9. Acknowledgements

We sincerely thank [MinerU](https://github.com/opendatalab/MinerU) for their outstanding work, whose powerful PDF/document text extraction capabilities provided essential support for our data loading process.
We also thank [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for offering an efficient and user-friendly framework for large model fine-tuning, which greatly facilitated rapid iteration in our training and experimentation workflows.
Our gratitude extends to all contributors in the open-source community—their efforts collectively drive the development of DataFlow.
We thank Zhongguancun Academy for their API and GPU support.


<a id="awesome-dataflow"></a>

## 🌟 10. Awesome Work Using DataFlow & DataFlow Ecosystem

This section highlights **projects, research works, and applications** built on top of DataFlow or deeply integrated with the DataFlow ecosystem.

**📌 Curated list of featured projects:**
[[Awesome Work Using DataFlow](./awesome_dataflow.md)]

We warmly welcome the community to contribute new entries via **Pull Requests**. 🙌 [Detailed Guidance](https://opendcai.github.io/DataFlow-Doc/en/guide/df_ecosystem/) can help you creating a Dataflow extension repository from DataFlow-CLI.




## 🤝 11. Community & Support

Join the DataFlow open-source community to ask questions, share ideas, and collaborate with other developers!

•	📮 [GitHub Issues](../../issues): Report bugs or suggest features

•	🔧 [GitHub Pull Requests](../../pulls): Contribute code improvements

•	💬 Join our community groups to connect with us and other contributors!

<div align="center">
  <img src="https://github.com/user-attachments/assets/8bc519c5-8ddb-4d34-b596-ae1b5c7cd20a" width="60%">
</div>

## 📜 12. Citation

If you use DataFlow in your research, feel free to give us a cite.

```bibtex
@article{liang2025dataflow,
  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
  journal={arXiv preprint arXiv:2512.16676},
  year={2025}
}
```

## 📊 13. Statistics

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
