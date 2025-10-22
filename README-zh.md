# DataFlow

<div align="center">
  <img src="https://github.com/user-attachments/assets/3fe636ad-3026-4faf-aa44-c84b8f97a05d">

[![Documents](https://img.shields.io/badge/官方文档-单击此处-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/DataFlow-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/DataFlow?style=social)](https://github.com/OpenDCAI/DataFlow)
[![](https://img.shields.io/github/issues-raw/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/issues)
[![](https://img.shields.io/github/contributors/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/DataFlow?color=green)](https://github.com/OpenDCAI/DataFlow)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/DataFlow)](https://github.com/OpenDCAI/DataFlow/commits/main/) -->

🎉 如果你认可我们的项目，欢迎在 GitHub 上点个 ⭐ Star，关注项目最新进展。

**新手友好的学习资源（持续更新中）**：🎬  [DataFlow 视频教程](https://space.bilibili.com/3546929239689711?spm_id_from=333.337.0.0)；📚 [DataFlow 图文教程](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)


简体中文 | [English](./README.md)

</div>

https://github.com/user-attachments/assets/bebe6e47-54bc-43da-abbc-a9c6a29ee44f

## 📰 1. 最新动态

- [2025-06-28] 🎉 我们全新发布的以数据为中心的系统**DataFlow**已开源 —— 敬请关注后续更新！

## 🔍 2. 项目概述

  ![dataflow_framework](https://github.com/user-attachments/assets/8a7c5259-dac7-4a44-b0e2-d099e75639c8)

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

### 📖 5.2 参考DataFlow项目文档

详细**使用说明**和**入门指南**，请参考我们的 [项目文档](https://OpenDCAI.github.io/DataFlow-Doc/)。

## 🧪 6. 实验结果

如需详细的实验设置，请参考文档或论文说明。

### 📝 6.1 文本流程（Text Pipeline）

#### 6.1.1 预训练数据过滤流程

我们将 `预训练数据处理流程` 应用于从 RedPajama 数据集中随机采样的数据，最终保留率为 **13.65%**。使用 `QuratingScorer` 进行质量评估，结果如下图所示：在**写作风格、专业性要求、事实准确性和教育价值**四个维度上，过滤后的数据显著优于原始数据，验证了 DataFlow 预训练数据处理流程的有效性。

<div align="center">
  <img src="https://github.com/user-attachments/assets/bc756c64-6640-4f46-b8ed-a4cd9be0a623" width="60%">
</div>



#### 6.1.2 微调（SFT）数据过滤流程

我们从 `alpaca` 数据集中筛选了 3000 条高质量数据，与随机选取的 3000 条 `alpaca` 数据进行对比，并在 Qwen2.5-7B 模型上进行 SFT 训练。对比结果如下：

<div align="center">
  <img src="https://github.com/user-attachments/assets/38d477d4-523d-4843-83f7-b7f518a18c1d" width="60%">
</div>

### 🧠 6.2 推理流程（Reasoning Pipeline）

我们在 Qwen2.5-32B-Instruct 模型上，使用 Reasoning Pipeline 合成的 1000 条和 5000 条数据进行了微调训练（SFT），评估其对模型推理能力的提升，结果如下图所示：

<div align="center">
  <img src="https://github.com/user-attachments/assets/d3af9728-0372-4c2c-9cd3-73f1e337d4c0" width="60%">
</div>

### 🗃️ 6.3 Text2SQL 流程

我们使用 DataFlow-Text2SQL 流程构建数据，并分别通过监督微调（SFT）与强化学习（RL）对 Qwen2.5-Coder-7B-Instruct 模型进行了训练。实验结果如下：

<div align="center">
  <img src="https://github.com/user-attachments/assets/7809f57a-33c5-4792-b91b-10e4f39bafc1" width="60%">
</div>


## 📄 7. 发表论文

我们团队已发表以下论文，并作为构成DataFlow系统的核心组件：

| 论文标题 | DataFlow组件 | 会议 | 年份 |
|---------|-------------|:------:|------|
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

## 🤝 10. 社区与支持

欢迎加入 DataFlow 开源社区，提出问题、分享想法、与其他开发者一起共建项目！

•	📮 [GitHub Issues](../../issues)：提交 Bug 或功能建议。

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进。

•	💬 欢迎扫码加入下方社群（微信群、小红书、Twitter），与我们和其他开发者互动交流~

<div align="center">
  <img src="https://github.com/user-attachments/assets/3c2e5d4d-d1ea-4d8c-9146-ff14e657e857" width="60%">
</div>


## 📜 11. 引用

如果 DataFlow 对你的研究或项目有帮助，欢迎引用支持我们：

```bibtex
@misc{dataflow2025,
  author       = {DataFlow Develop Team},
  title        = {DataFlow: A Unified Framework for Data-Centric AI},
  year         = {2025},
  howpublished = {\url{https://github.com/OpenDCAI/DataFlow}},
  note         = {Accessed: 2025-07-08}
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
