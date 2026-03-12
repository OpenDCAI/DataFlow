# DataFlow

<div align="center">

**大模型数据生成、清洗与准备，一站式搞定**

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


[![Discord Online](https://img.shields.io/discord/1479323317096939551?logo=discord&label=discord&color=%235966F0)](https://discord.gg/e4mKEaFptu)
[![wechat](https://img.shields.io/badge/微信-brightgreen?logo=wechat&logoColor=white)](https://github.com/user-attachments/assets/3c2e5d4d-d1ea-4d8c-9146-ff14e657e857)

<a href="https://trendshift.io/repositories/16045" target="_blank"><img src="https://trendshift.io/api/badge/repositories/16045" alt="OpenDCAI%2FDataFlow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

可视化、低代码流水线，支持跨领域和用例的灵活编排。💪

将原始数据转化为高质量的 LLM 训练数据集。🔧

🎉 以更低的成本获得更智能的 LLM —— 在 GitHub 上给我们点个Star ⭐ 以获取最新更新。

**初学者友好学习资源（持续更新）**:
[[🎬 视频教程]](https://space.bilibili.com/3546929239689711?spm_id_from=333.337.0.0)
[[📚 文字教程]](https://wcny4qa9krto.feishu.cn/wiki/I9tbw2qnBi0lEakmmAGclTysnFd)

简体中文 | [English](./README.md)

<img src="https://github.com/user-attachments/assets/6d8fd795-7f5b-4c45-b14d-5bbe6bf99766" />
</div>

## 📰 1. 新闻

* **[2026-02-02] 🖥️ DataFlow WebUI 正式发布！**
  通过一条命令 `dataflow webui` 即可启动可视化流水线构建器，在直观的网页界面中构建并运行 DataFlow 流水线。👉 [WebUI 文档](#54-webui)
  <div style="display: flex; gap: 12px;">
    <img src="https://github.com/user-attachments/assets/b4f172d6-7753-4121-b981-55046a7a9e43" width="45%" />
    <img src="https://github.com/user-attachments/assets/b2147987-3b1e-4f56-9818-3d5e7440fa58" width="45%" />
  </div>
* **[2026-01-20] 🌟 DataFlow Awesome Works 上线！**
  新增板块用于展示基于 DataFlow 的开源项目与研究工作，欢迎提交 Pull Request 分享你的成果！👉 [Awesome Works](#awesome-dataflow)

* **[2025-12-19] 🎉 DataFlow 技术报告正式发布！**
  欢迎阅读并引用我们的 arXiv 论文：[https://arxiv.org/abs/2512.16676](https://arxiv.org/abs/2512.16676)

* **[2025-11-20] 🤖 DataFlow 全新 Data Agents 发布！**
  现在即可体验，并通过 Bilibili 教程快速上手：[https://space.bilibili.com/3546929239689711/lists/6761342?type=season](https://space.bilibili.com/3546929239689711/lists/6761342?type=season)

* **[2025-06-28] 🎉 DataFlow 正式开源发布！** 我们全新发布的以数据为中心的系统**DataFlow**已开源 —— 敬请关注后续更新！

## 🔍 2. 什么是 DataFlow？

DataFlow 是一个专门为从嘈杂来源（PDF、纯文本、低质量 QA）中**生成、精炼、评估和过滤**高质量 AI 数据而设计的数据准备和训练系统，从而通过针对性训练（预训练、指令微调 SFT、强化学习训练 RL）或 RAG 系统，提升大语言模型 (LLM) 在医疗、金融、法律和学术研究等特定领域的性能。

通过“基于算子 (operator-based)”的设计，DataFlow 将整个数据清洗工作流转化为可重现、可重用且可共享的“流水线 (pipeline)”，为以数据为中心的 AI 社区提供核心基础设施。此外，我们开发了智能化的 `DataFlow-agent`，能够根据需求通过重新组合现有算子或创建新算子来动态组装新的“流水线”。

![df_overview_final_300](https://github.com/user-attachments/assets/57dd0838-6e24-4814-a89a-02ca0667bd5c)


## 🔍 2. 核心特性

### ✅ 2.1 开箱即用的数据合成与清洗流水线

* 高质量训练数据生成
  * 文本、数学和代码数据生成（参见 DataFlow-Instruct-10K 结果）
  * 通过 AgenticRAG 和 Text2SQL 等工具进行数据生成


* 结构化数据提取
  * 大规模 PDF → QA 转换
  * 大规模书籍 PDF → 视觉 QA 转换


* 科学数据工作流管理
  * Text2SQL 工作流管理（被 ICDE 2026 录用）
  * 数学数据工作流（被 KDD 2026 录用）

### ⚙️ 2.2 灵活的自定义流水线编排

* 10+ 核心算子定义了交互模式和设计原则
* 100+ 特定流水线算子可供重用或参考
* 全面支持创建自定义算子 —— 插件式设计，可通过 GitHub 或 PyPI 轻松封装和分发

### 🧠 2.3 可复现、可复用且可分发的以数据为中心的 AI 系统

* 数据治理算法被封装为算子流水线，实现了不同数据治理策略的可复现性和公平比较（❤️ 研究友好）
* 轻松更换底层大模型复用原有管线，快速分析模型性能与数据质量之间的关系
* 基于 Python 和 Git 生态系统，方便分发、管理和追溯高质量、**用户定义**的数据治理算子和流水线（❤️ 企业友好）

## 🛠️ 3. DataFlow 套件

DataFlow 套件提供了基本的算力设施，以配合 DataFlow 主仓库实现 LLM 数据准备的自动化和规模化。它由四个紧密集成的层组成：

* [DataFlow-WebUI](#️54-webui) – 一个直观的可视化界面，用于通过拖拽式算子工作流构建和管理复杂的数据流水线。
* [DataFlow-Agent](https://github.com/OpenDCAI/DataFlow-Agent) – 一个由 AI 驱动的助手，根据用户的高层意图动态组合、执行和优化算子及流水线。
* [DataFlow-Ecosystem](#awesome-dataflow) – 一个标准化的算子注册模块化分发层。它允许特定领域的模块（例如 [DataFlow-MM](https://github.com/OpenDCAI/DataFlow-MM)、DataFlow-AI4S）在统一的抽象下贡献可扩展库。
* [RayOrch](https://github.com/OpenDCAI/RayOrch) – 一个基于 Ray 构建的高性能编排层，为大规模数据任务提供分布式计算调度和资源管理。

这些组件共同构成了一个统一、可扩展的环境，将原始数据转化为模型就绪的智能信息。

## 4. 为什么使用 DataFlow？

数据生成和清洗对于高质量模型至关重要，但对于企业和个人而言，这些任务往往耗时、耗力且成本高昂。**DataFlow 提供了一站式解决方案，高效应对这些挑战。**
与 Nemo-Curator 和 Data-Juicer 等系统相比，DataFlow 提供：

* **更强的数据合成模块支持** – 无缝集成文本、代码和数学数据生成流水线，用于高质量训练数据集。
* **类 PyTorch 的编程管理** – 清晰的 **流水线 (Pipeline) → 算子 (Operator) → 提示词 (Prompt)** 分层结构，用于工作流控制。
* **原则化和多类别的算子分类** – 算子被系统地组织到诸如**生成、评估、过滤和精炼**等多个功能类别中，形成了科学的多维分类法，反映了数据准备的不同阶段，并支持精确的算子选择与组合。
* **用户友好设计，易于调试和上手** – 简化的工作流模式降低了学习曲线，加速了实验进程。

## 🔧 5. 算子如何工作？

DataFlow 算子的设计秉持**简单与清晰**的原则。

算子接收结构化输入（JSON, JSONL, CSV），经过智能化处理后产生高质量输出。
每个算子封装了一个特定的数据处理任务，提供简洁且一致的 API，易于理解和集成。类 PyTorch 的设计使其直观且开箱即用，让您可以快速构建、组合和自定义流水线，无需处理复杂的模板代码。

更多详情请参考 [算子文档](https://opendcai.github.io/DataFlow-Doc/zh/api/home/)。下面是一个演示如何调用 `PromptedGenerator` 算子的极简示例：

示例输入数据 (json/jsonl 样式):

```json
// input.json
[
  {"problem": "What is 17 + 25?"},
  {"problem": "If x = 3, compute 2x^2 + 1."}
]

```

算子调用代码：

```python
from dataflow.operators.core_text import PromptedGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

# 将输入文件设置到全局存储类
storage = FileStorage(first_entry_file_name="./input.json",)

# 配置 LLM 服务（例如 OpenAI API）
# api key 需要通过 `export DF_API_KEY=sk-xxx` 设置
llm_serving = APILLMServing_request(
    api_url="https://api.openai.com/v1/chat/completions",
)

prompted_generator = PromptedGenerator(
    llm_serving=llm_serving,  # 预配置的 LLM 后端
    system_prompt="Please solve this math problem."
)

prompted_generator.run(
    storage=self.storage.step(),  # 数据管理（细节省略）
    input_key="problem",          # 从该列读取
    output_key="solution"         # 写入该列
)

```

运行后，算子会将生成的结果追加到 output_key 中。例如，输出数据 (json/jsonl 样式) 变为：

```json
// dataflow_step1.json
[
    {"problem":"What is 17 + 25?","solution":"42"},
    {"problem":"If x = 3, compute 2x^2 + 1.","solution":"19"}
]

```

<details>
<summary><h2>🛠️ 6. 流水线 (点击展开)</h2></summary>

### 🔧 6.1 开箱即用的流水线

DataFlow 目前包含的流水线如下：
- [📝 **文本处理流水线（Text Pipeline）**](https://opendcai.github.io/DataFlow-Doc/zh/guide/textpipeline)：从大规模纯文本（多为网络爬取）中挖掘问答对，用于监督微调和强化学习训练。
  - ![dataflow_text_pipeline](https://github.com/user-attachments/assets/34e3aef2-ba4f-4997-9127-9d21fdb2dede)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text)

- [🧠 **推理流水线（Reasoning Pipeline）**](https://opendcai.github.io/DataFlow-Doc/zh/guide/reasoningpipeline/#_2-question-handling)：增强已有问答对，添加 (1) 长链式推理（Chain-of-Thought），(2) 类别标注，(3) 难度估计。
  - ![dataflow_reasoning_pipeline](https://github.com/user-attachments/assets/fef5829b-3991-4dcb-99ad-d61d95c982ea)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Reasonning)

- [🗃️ **Text2SQL 流水线**](https://opendcai.github.io/DataFlow-Doc/zh/guide/text2sqlpipeline/)：将自然语言问题转化为 SQL 查询，辅以解释、思维链推理和数据库结构上下文信息。
  - ![dataflow_text2sql_pipeline](https://github.com/user-attachments/assets/bae9914e-851b-4502-8696-291d6c1b8824)
  - [[HuggingFace🤗 示例数据]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Text2SQL)

- [📚 **知识库清洗流水线**](https://opendcai.github.io/DataFlow-Doc/zh/guide/r51ooua8/)：从表格、PDF 和 Word 文档等非结构化数据源中提取并整理知识，将其转化为可用于下游 RAG 或 QA 配对生成的可用条目。
  - ![dataflow_KnowledgeBaseClean_pipeline](https://github.com/user-attachments/assets/6f21e682-ec10-42af-b5e2-8fec2929eeae)

- [🤖 **Agent式RAG流水线**](https://opendcai.github.io/DataFlow-Doc/zh/guide/agenticrag_pipeline/)：从已有问答或知识库中挖掘需要外部知识才能作答的问答对，用于训练 Agentic RAG 模型。
  - ![dataflow_agenticRAG_pipeline](https://github.com/user-attachments/assets/65e80dca-f286-495b-abb7-804b3fc34a53)




### ⚙️ 6.2 灵活的算子流水线

在此框架下，算子被分为基础算子、通用算子、特定领域算子和评估算子等，支持数据处理和评估功能。详情请参考 [文档](https://OpenDCAI.github.io/DataFlow-Doc/)。

### 🤖 6.3 智能体引导的流水线

* **DataFlow Agent**: 一个智能助手，能够进行数据分析，编写自定义“算子”，并根据特定的任务目标自动将它们编排成“流水线”。
* [[HuggingFace🤗 **DataFlow Agent** 演示输入与输出]](https://huggingface.co/datasets/Open-Dataflow/dataflow-demo-Agent)



</details>

## ⚡ 7. 快速开始

### 🛠️ 7.1 环境配置与安装

> DataFlow 支持 Python >= 3.10 环境，已在 Windows、Linux 和 MacOS 上的 Python 3.10, 3.11, 3.12 版本通过测试。

请使用以下命令进行环境配置与安装👇

我们推荐使用 [uv](https://docs.astral.sh/uv/) 安装 DataFlow 以加速下载。

```shell
pip install uv
uv pip install open-dataflow

```

如果您想使用自己的 GPU 进行本地推理，请使用：

```shell
pip install uv
uv pip install open-dataflow[vllm]

```

安装完成后，可以使用以下命令检查 DataFlow 是否安装正确：

```shell
dataflow -v

```

如果安装成功，您将看到：

```log
open-dataflow codebase version: 1.0.0
        Checking for updates...
        Local version:  1.0.0
        PyPI newest version:  1.0.0
You are using the latest version: 1.0.0.

```

#### 🐳 7.2 Docker 安装（备选）

我们也提供了 **Dockerfile** 以便部署，并提供 **预构建的 Docker 镜像** 以便立即使用。

##### 选项 1：使用预构建 Docker 镜像

您可以直接拉取并使用我们的预构建 Docker 镜像：

```shell
# 拉取预构建镜像
docker pull molyheci/dataflow:cu124

# 运行带有 GPU 支持的容器
docker run --gpus all -it molyheci/dataflow:cu124

# 在容器内验证安装
dataflow -v
```

##### 选项 2：从 Dockerfile 构建

或者，您可以根据提供的 Dockerfile 构建 Docker 镜像：

```shell
# 克隆仓库 (HTTPS)
git clone https://github.com/OpenDCAI/DataFlow.git
# 或使用 SSH
# git clone git@github.com:OpenDCAI/DataFlow.git

cd DataFlow

# 构建 Docker 镜像
docker build -t dataflow:custom .

# 运行容器
docker run --gpus all -it dataflow:custom

# 在容器内验证安装
dataflow -v

```

> **注意**: Docker 镜像包含 CUDA 12.4.1 支持，并预装了 vLLM 以实现 GPU 加速。请确保您已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 以使用 GPU 功能。

### 🚀 7.3 通过 Google Colab 快速开始

您可以直接在 Google Colab 上开始您的第一个 DataFlow 翻译项目。
按照提供的指南，您可以无缝地从简单的翻译示例扩展到更复杂的 DataFlow 流水线。

👉 [通过 Google Colab 开启 DataFlow](https://colab.research.google.com/drive/1haosl2QS4N4HM7u7HvSsz_MnLabxexXl?usp=sharing)

### 📖 7.3 参考项目文档

有关详细的**使用说明**和**入门指南**，请访问我们的 [DataFlow 文档](https://OpenDCAI.github.io/DataFlow-Doc/)。

<a id="54-webui"></a>

### 🖥️ 7.4 DataFlow-WebUI

DataFlow 提供了一个**基于 Web 的 UI (WebUI)**，用于可视化流水线的构建与执行。

<div style="display: flex; gap: 12px;">
  <img src="https://github.com/user-attachments/assets/b4f172d6-7753-4121-b981-55046a7a9e43" width="45%" />
  <img src="https://github.com/user-attachments/assets/b2147987-3b1e-4f56-9818-3d5e7440fa58" width="45%" />
</div>

安装 DataFlow 主仓库后，只需运行以下命令即可启动 `DataFlow-WebUI`：

```bash
dataflow webui

```

这将自动下载并启动最新的 **DataFlow-WebUI** 并在浏览器中打开（如果未自动打开，请访问 `http://localhost:8000/`）。

#### 📚 7.4.1 WebUI 文档

* 中文: [DataFlow-WebUI 文档: https://wcny4qa9krto.feishu.cn/wiki/F4PDw76uDiOG42k76gGc6FaBnod](https://wcny4qa9krto.feishu.cn/wiki/F4PDw76uDiOG42k76gGc6FaBnod)
* 英文: [DataFlow-WebUI 文档: https://wcny4qa9krto.feishu.cn/wiki/SYELwZhh9ixcNwkNRnhcLGmWnEg](https://wcny4qa9krto.feishu.cn/wiki/SYELwZhh9ixcNwkNRnhcLGmWnEg)

#### 🛠️ 7.4.2 开发仓库

* [https://github.com/OpenDCAI/DataFlow-webui](https://github.com/OpenDCAI/DataFlow-webui)

## 🧪 8. 实验结果

### 8.1 DataFlow-Instruct-10k

**DataFlow-Instruct-10K** 是一个由 DataFlow 框架生成的统一多领域指令数据集。它通过跨越数学推理、代码和通用文本指令的多个自动化数据准备流水线构建。每个流水线遵循“生成-评估-过滤-精炼”工作流，以合成和策划高质量的“指令-响应”对。最终的数据集包含约 10K 个样本，为指令微调提供高质量监督，使基座模型能够以更少的训练样本达到全量训练指令模型的性能水平。

有关详细的实验设置，请访问我们的 [DataFlow 技术报告](https://arxiv.org/abs/2512.16676)。

| 模型 | 数学平均 (Math-Avg) | 代码平均 (Code-Avg) | 知识平均 (Knowledge-Avg) |
| --- | --- | --- | --- |
| **Qwen2-7B 系列** |  |  |  |
| Base | 20.1 | 66.3 | 76.2 |
| + Infinity-Instruct-10K | 29.0 | 67.8 | 76.2 |
| + Infinity-Instruct-1M | 27.9 | **68.2** | **76.2** |
| + **DataFlow-Instruct-10K** | **32.4** | 66.2 | 76.1 |
| **Qwen2.5-7B 系列** |  |  |  |
| Base | 37.1 | 76.5 | 76.0 |
| + Infinity-Instruct-10K | 22.6 | 77.6 | 75.8 |
| + Infinity-Instruct-1M | 33.3 | 78.0 | 75.8 |
| + **DataFlow-Instruct-10K** | **46.7** | **78.6** | **76.2** |

<details>
<summary><h3>🛠️ 8.2 其他流水线结果 (点击展开)</h3></summary>

#### 8.2.1 文本流水线

##### 8.2.1.1 预训练数据过滤流水线

从 SlimPajama-627B 语料库中，我们提取了一个 100B-token 子集，并应用了多个 DataFlow 文本预训练过滤器。我们使用 Megatron-DeepSpeed 框架从头开始训练了一个 Qwen2.5-0.5B 模型，训练量为 30B tokens，结果如下：

| 方法 | ARC-C | ARC-E | MMLU | HellaSwag | WinoGrande | Gaokao-MathQA | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Random-30B** | 25.26 | 43.94 | 27.03 | 37.02 | 50.99 | 27.35 | 35.26 |
| **Qurating-30B** | 25.00 | 43.14 | 27.50 | 37.03 | 50.67 | 26.78 | 35.02 |
| **FineWeb-Edu-30B** | 26.45 | 45.41 | 27.41 | 38.06 | 50.43 | 25.64 | 35.57 |
| **DataFlow-30B** | 25.51 | 45.58 | 27.42 | 37.58 | 50.67 | 27.35 | **35.69** |

##### 8.2.1.2 SFT 数据过滤与合成流水线

为了研究小规模 SFT 数据质量，我们使用 LLaMA-Factory 在 WizardLM 和 Alpaca 数据集上对 Qwen2.5-7B 基座模型进行了微调。对于每个数据集，我们将 5K 个随机采样实例的集合与 5K 个经过 DataFlow SFT 流水线过滤的实例集合进行了比较。此外，我们使用 DataFlow 的 Condor Generator 和 Condor Refiner 流水线合成了一个 15k 大小的数据集 DataFlow-SFT-15K，随后通过 DataFlow 的 SFT 过滤流水线（不包括 Instagram 过滤器）。基准测试包括全面的数学、代码和知识评估套件。

#### 8.2.2 数学基准测试

| 方法 | math | gsm8k | aime24 | minerva | olympiad | 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| **Alpaca (random)** | 54.9 | 77.2 | 13.3 | 14.0 | 27.0 | 37.3 |
| **Alpaca (filtered)** | 60.3 | 80.0 | 13.3 | 14.7 | 30.7 | 39.8 |
| **WizardLM (random)** | 61.1 | 84.2 | 6.7 | 18.0 | 29.3 | 39.9 |
| **WizardLM (filtered)** | 69.7 | 88.8 | 10.0 | 19.9 | 35.4 | 44.8 |
| **DataFlow-SFT-15K (random)** | 72.6 | 89.6 | 13.3 | 37.9 | 32.9 | **49.3** |
| **DataFlow-SFT-15K (filtered)** | 73.3 | 90.2 | 13.3 | 36.0 | 35.9 | **49.7** |

#### 8.2.3 代码基准测试

| 方法 | HumanEval | MBPP | 平均 |
| --- | --- | --- | --- |
| **Alpaca (random)** | 71.3 | 75.9 | 73.6 |
| **Alpaca (filtered)** | 73.8 | 75.7 | 74.8 |
| **WizardLM (random)** | 75.6 | 82.0 | **78.8** |
| **WizardLM (filtered)** | 77.4 | 80.4 | **78.9** |
| **DataFlow-SFT-15K (random)** | 79.9 | 75.9 | 77.9 |
| **DataFlow-SFT-15K (filtered)** | 82.9 | 74.9 | **78.9** |

#### 8.2.4 知识基准测试

| 方法 | MMLU | C-EVAL | 平均 |
| --- | --- | --- | --- |
| **Alpaca (random)** | 71.8 | 80.0 | 75.9 |
| **Alpaca (filtered)** | 71.8 | 80.0 | 75.9 |
| **WizardLM (random)** | 71.8 | 79.2 | 75.5 |
| **WizardLM (filtered)** | 71.9 | 79.6 | 75.8 |
| **DataFlow-SFT-15K (random)** | 72.1 | 80.0 | **76.1** |
| **DataFlow-SFT-15K (filtered)** | 72.2 | 80.4 | **76.3** |

#### 8.2.5 对话合成流水线

我们使用 DataFlow 的对话生成流水线合成了 DataFlow-Chat-15K，并在其上微调了 Qwen2.5-7B-Base。基准包括 ShareGPT-15K、UltraChat-15K 及其全量版本。我们在特定领域任务 (TopDial, Light) 和通用基准 (MMLU, AlpacaEval, Arena-Hard) 上进行了评估。

##### 8.2.5.1 对话基准测试

| 模型 | TopDial | Light | 平均 |
| --- | --- | --- | --- |
| **Qwen2.5-7B** | 7.71 | 7.79 | 7.75 |
| **+ ShareGPT-15K** | 7.75 | 6.72 | 7.24 |
| **+ UltraChat-15K** | 7.72 | 6.83 | 7.28 |
| **+ DataFlow-Chat-15K** | **7.98** | **8.10** | **8.04** |

##### 8.2.5.2 通用基准测试

| 模型 | MMLU | AlpacaEval | Arena-Hard | 平均 |
| --- | --- | --- | --- | --- |
| **Qwen2.5-7B** | 71.45 | 7.05 | 0.60 | 26.36 |
| **+ ShareGPT-15K** | 73.09 | 3.70 | 1.30 | 26.03 |
| **+ UltraChat-15K** | 72.97 | 3.97 | 0.80 | 25.91 |
| **+ DataFlow-Chat-15K** | 73.41 | **10.11** | 1.10 | **28.21** |

#### 8.2.6 推理流水线

我们采用 NuminaMath 数据集作为高质量种子集。我们比较了三种训练源：(1) Open-R1 的随机 10K 子集，(2) Synthetic-1 的随机 10K 子集，以及 (3) 我们使用 DataFlow 构建的 10K 合成 DataFlow-Reasoning-10K 数据集。

| 设置 | 模型 | gsm8k | math | amc23 | olympiad | gaokao24_mix | minerva | AIME24@32 | AIME25@32 | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | **Qwen2.5-32B-Instruct** | 95.8 | 73.5 | 70.0 | 38.5 | 42.9 | 26.5 | 16.8 | 11.6 | 46.95 |
| 1 Epoch | **+ SYNTHETIC-1-10k** | 92.9 | 71.8 | 52.5 | 38.4 | 23.1 | 24.3 | 35.6 | 34.0 | 46.6 |
| 1 Epoch | **+ Open-R1-10k** | 91.5 | 72.3 | 65.0 | 38.4 | 20.9 | 24.6 | 43.0 | 33.5 | 48.7 |
| 1 Epoch | **+ DataFlow-Reasoning-10K** | 93.9 | 72.3 | 72.5 | 38.7 | 38.5 | 26.5 | 35.9 | 34.5 | **51.6** |
| 2 Epochs | **+ SYNTHETIC-1-10k** | 94.5 | 78.4 | 75.0 | 45.0 | 24.2 | 28.3 | 48.4 | 37.9 | 54.0 |
| 2 Epochs | **+ Open-R1-10k** | 93.9 | 77.2 | 80.0 | 44.1 | 20.9 | 25.4 | 51.0 | 40.7 | 54.2 |
| 2 Epochs | **+ DataFlow-Reasoning-10K** | 94.4 | 76.6 | 75.0 | 45.2 | 42.9 | 25.7 | 45.4 | 40.0 | **55.7** |

#### 8.2.7 代码流水线

我们从 Ling-Coder-SFT 语料库中随机抽取了 20k 实例，并通过 DataFlow 代码流水线进行处理。这产生了三个不同规模的精选代码指令数据集：DataFlow-Code-1K, DataFlow-Code-5K, 和 DataFlow-Code-10K，每个数据集旨在为代码生成任务提供高质量、经过流水线精炼的监督信号。我们将我们的合成数据集与 Code-Alpaca-1k 和 Self-OSS-Instruct-SC2-Exec-Filter-1k 进行了对比。

##### 8.2.7.1 在 Qwen2.5-7B-Instruct 上训练

| 训练数据 | BigCodeBench | LiveCodeBench (v6) | CruxEval (I) | CruxEval (O) | HumanEval+ | 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen2.5-7B-Instruct** | 35.3 | 23.4 | 44.8 | 43.9 | 72.6 | 44.0 |
| **+ Code Alpaca-1K** | 33.3 | 18.7 | 45.6 | 46.4 | 66.5 | 42.1 |
| **+ Self-OSS** | 31.9 | 21.4 | 46.9 | 45.9 | 70.1 | 43.2 |
| **+ DataFlow-Code-1K** | 35.5 | 25.7 | 48.0 | 45.1 | 72.6 | 45.4 |
| **+ DataFlow-Code-5K** | 36.2 | **26.4** | 48.6 | 45.0 | 73.2 | 45.9 |
| **+ DataFlow-Code-10K** | **36.8** | 26.0 | **48.8** | **45.4** | **73.8** | **46.2** |

##### 8.2.7.2 在 Qwen2.5-14B-Instruct 上训练

| 训练数据 | BigCodeBench | LiveCodeBench (v6) | CruxEval (I) | CruxEval (O) | HumanEval+ | 平均 |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen2.5-14B-Instruct** | 37.5 | 33.4 | 48.0 | 48.5 | 74.4 | 48.4 |
| **+ Code Alpaca-1K** | 37.0 | 28.2 | 50.2 | 49.6 | 71.3 | 47.3 |
| **+ Self-OSS** | 36.9 | 22.3 | 52.6 | 50.1 | 68.3 | 46.0 |
| **+ DataFlow-Code-1K** | 41.4 | **33.7** | 51.0 | 50.9 | **77.3** | 50.9 |
| **+ DataFlow-Code-5K** | 41.1 | 33.2 | 52.5 | 50.6 | 76.2 | 50.7 |
| **+ DataFlow-Code-10K** | **41.9** | 33.2 | **52.9** | **51.0** | 76.2 | **51.0** |

</details>

## 📄 9. 出版物

我们的团队发表了以下论文，构成了 DataFlow 系统的核心组件：

| 论文题目 | DataFlow 组件 | 会议/期刊 | 年份 |
| --- | --- | --- | --- |
| [Text2SQL-Flow: A Robust SQL-Aware Data Augmentation Framework for Text-to-SQL](https://arxiv.org/abs/2505.13903) | Text2SQL 数据增强 | ICDE | 2026 |
| [Let's Verify Math Questions Step by Step](https://arxiv.org/abs/2505.13903) | 数学问题质量评估 | KDD | 2026 |
| [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](https://arxiv.org/pdf/2502.13383) | 用于数据处理和评估的多模态推理验证框架 | ACL | 2025 |
| [Efficient Pretraining Data Selection for Language Models via Multi-Actor Collaboration](https://arxiv.org/pdf/2410.08102) | 多角色协同数据选择机制，用于增强数据过滤和处理 | ACL | 2025 |

**贡献机构**:
<img src="./static/logo/pku.png" alt="PKU" height="30"/>
<img src="./static/logo/hkust.png" alt="HKUST" height="30"/>
<img src="./static/logo/CAS.png" alt="CAS" height="30"/>
<img src="./static/logo/shanghai_ailab.png" alt="Shanghai AI Lab" height="30"/>
<img src="./static/logo/baichuan.png" alt="Baichuan" height="30"/>
<img src="./static/logo/ant_group.png" alt="Ant Group" height="30"/>

## 🏆 10. 奖项与成就

我们荣幸地在两项重大国际 AI 竞赛中获得**第一名**，这彰显了 DataFlow 及其推理能力的卓越性与稳健性：

| 竞赛项目 | 赛道 | 奖项 | 组织者 | 日期 |
| --- | --- | --- | --- | --- |
| **ICML 2025 自动化数学推理及其扩展挑战赛** | 赛道 2：*结合图表和表达式的物理推理* | 🥇**冠军 (First Place)** | ICML AI for Math Workshop & AWS Codabench | 2025年7月18日 |
| **2025 语言与智能技术竞赛 (LIC)** | 赛道 2：*北京人工智能研究院* | 🥇**一等奖** | 北京人工智能研究院 (BAAI) & 百度 | 2025年8月10日 |

<div align="center">

<table>
<tr>
<td align="center" width="50%">
<img src="https://github.com/user-attachments/assets/8f28e0fe-c883-42c0-b224-3693f6281a14" alt="ICML 2025 Certificate" width="95%">




<sub><em>ICML 2025 自动化数学推理挑战赛 —— 冠军</em></sub>
</td>
<td align="center" width="30%">
<img src="https://github.com/user-attachments/assets/364618b6-4dfa-4c34-928f-e3da85cbd03a" alt="LIC 2025 Certificate" width="95%">




<sub><em>BAAI 2025 语言与智能技术竞赛 —— 一等奖</em></sub>
</td>
</tr>
</table>

</div>

<a id="awesome-dataflow"></a>

## 🌟 11. 使用 DataFlow 的优秀作品与 DataFlow 生态

本板块重点展示基于 DataFlow 构建或深度集成于 DataFlow 生态系统的**项目、研究成果和应用**。

📌 **精选项目列表**：
[[Awesome Work Using DataFlow](./awesome_dataflow.md)]

我们热忱欢迎社区通过 **Pull Requests** 贡献新条目。🙌 [详细指南](https://opendcai.github.io/DataFlow-Doc/en/guide/df_ecosystem/) 可以帮助您通过 DataFlow-CLI 创建 DataFlow 扩展仓库。

## 💐 12. 致谢

我们衷心感谢 [MinerU](https://github.com/opendatalab/MinerU) 的杰出工作，其强大的 PDF/文档文本提取能力为我们的数据加载过程提供了重要支持。
我们也要感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，它提供了一个高效且用户友好的大模型微调框架，极大地方便了我们训练和实验工作流的快速迭代。
我们要感谢开源社区的所有贡献者 —— 他们的努力共同推动了 DataFlow 的发展。
感谢中关村研究院提供的 API 和 GPU 支持。

## 🤝 13. 社区与支持

加入 DataFlow 开源社区，提出问题、分享想法并与其他开发者合作！

•	📮 [GitHub Issues](../../issues)：提交 Bug 或功能建议。

•	🔧 [GitHub Pull Requests](../../pulls)：贡献代码改进。

•	💬 加入我们的社区群组，与我们及其他贡献者建立联系！

<div align="center">
  <img src="https://github.com/user-attachments/assets/52febf13-5288-4bcd-95e8-9126dffbc409" width="60%">
</div>

## 📜 14. 引用

如果您在研究中使用了 DataFlow，请随时引用我们。

```bibtex
@article{liang2025dataflow,
  title={DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI},
  author={Liang, Hao and Ma, Xiaochen and Liu, Zhou and Wong, Zhen Hao and Zhao, Zhengyang and Meng, Zimo and He, Runming and Shen, Chengyu and Cai, Qifeng and Han, Zhaoyang and others},
  journal={arXiv preprint arXiv:2512.16676},
  year={2025}
}
```

<div align="center">
<sub>
关注
<a href="[https://zwt233.github.io/](https://zwt233.github.io/)" target="_blank"><strong>PKU-DCAI 研究团队</strong></a>
的小红书：<strong>26133106768</strong>
</sub>
</div>