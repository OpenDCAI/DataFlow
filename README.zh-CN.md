[English Readme](./README.md)

# DataFlow-Eval

<p align="center">
  <img src="./static/images/Face.png">
</p>
<a href="https://opensource.org/license/apache-2-0" target="_blank">
    <img alt="License: apache-2-0" src="https://img.shields.io/github/license/saltstack/salt" />
</a>
<a href="https://github.com/GAIR-NLP/ProX" target="_blank">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Open-DataFlow/Open-DataFlow-Eval?style=social" />
</a>
<a href="https://github.com/GAIR-NLP/ProX/issues" target="_blank">
    <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/Open-DataFlow/Open-DataFlow-Eval" />
</a>

DataFlow-Eval 是一个数据质量评估系统，可以从多个维度评估数据质量。我们主要支持具有强大理论支持的学术论文中的最新算法。

我们目前支持文本、图像、视频和多模态数据类型。

## Table of Contents
- [DataFlow-Eval](#dataflow-eval)
  - [Table of Contents](#table-of-contents)
  - [模块和模态支持](#模块和模态支持)
  - [🔥 新闻](#新闻)
  - [安装](#安装)
  - [快速开始](#快速开始)
    - [文本示例](#文本示例)
    - [图像示例](#图像示例)
    - [视频示例](#视频示例)
  - [数据评估文档](#数据评估文档)
    - [文本文档](#文本文档)
    - [图像文档](#图像文档)
    - [视频文档](#视频文档)
  - [数据评估算法](#数据评估算法)
    - [文本算法](#文本算法)
    - [图像算法](#图像算法)
    - [视频算法](#视频算法)
  - [👋 数据评估论文总结(综述)](#数据评估论文总结综述)

## 模块和模态支持

| 模块\模态    | 文本  | 图像  | 视频  | 图像-文本对 | 视频-文本对 |
| -------- | --- | --- | --- | ------ | ------ |
| **数据评估** | ✅   | ✅   | ✅   | ✅      | ✅      |


## 🔥新闻

- [2024-10-14] 🎉 我们在 [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)总结了数据评估相关论文

- [2024-10-14] 🎉 我们的数据评估系统开源了

## 安装

您可以用如下命令配置conda环境
```
conda create -n dataflow python=3.9

conda activate dataflow

pip install -e .
```

  
如果您想评估单个模态的数据，可以使用下面的安装代码👇

<details>
<summary>
<b>text data eval</b>
</summary>
<p>

```bash
pip install -e .[text]
pip install flash-attn==2.6.3
python -m spacy download en_core_web_sm
```

</p>
</details>

<details>
<summary>
<b>image data eval</b>
</summary>
<p>

```bash
pip install -e .[image]
pip install pyiqa==0.1.12
pip install transformers==4.44.2
```

</p>
</details>


<details>
<summary>
<b>video data eval</b>
</summary>
<p>

```bash
pip install -e .[video]
```
When evaluating video-caption data, please run the following command to install modified CLIP for EMScore:
```
pip install git+https://github.com/MOLYHECI/CLIP.git
```

</p>
</details>

<details>
<summary>
<b>All dependencies</b>
</summary>
<p>

```bash
pip install -e .[all]
pip install flash-attn==2.6.3
pip install pyiqa==0.1.12
pip install transformers==4.44.2
```

</p>
</details>

请参考[数据评估文档](#数据评估文档)查看参数的使用规则. 仅使用yaml参数便可以完成数据评估：

```
python test.py --config [your config file]
```
<p align="center">
  <img src="./static/images/example_1.png">
</p>
  
## 🚀快速开始

### 文本示例

- [Text Evaluation Demo](./demos/text_eval/text_eval_example.ipynb)
- [文本评估示例](./demos/text_eval/text_eval_example.zh-CN.ipynb)

### 图像示例
- [Image Evaluation Demo](./demos/image_eval/image_eval_example.ipynb)
- [图片评估示例](./demos/image_eval/image_eval_example.zh-CN.ipynb)

### 视频示例
- [Video Evaluation Demo](./demos/video_eval/video_eval_example.ipynb)
- [视频评估示例](./demos/video_eval/video_eval_example.zh-CN.ipynb)

## 数据评估文档

请参照下面的文档了解不同模态的数据评估👇

### 文本文档

- [Text Data Evaluation User Documentation (English)](./dataflow/Eval/Text/README.md)
- [文本数据评估使用文档 (中文)](./dataflow/Eval/Text/README.zh-CN.md)

### 图像文档

- [Image Data Evaluation User Documentation (English)](./dataflow/Eval/image/README.md)
- [图像数据评估使用文档 (中文)](./dataflow/Eval/image/README.zh-CN.md)

### 视频文档

- [Video Data Evaluation User Documentation (English)](./dataflow/Eval/video/README.md)
- [视频数据评估使用文档 (中文)](./dataflow/Eval/video/README.zh-CN.md)

## 数据评估算法

我们在这里总结了目前领先的数据评估算法

### 文本算法

- [Text Evaluation Algorithm Document (English)](./docs/text_metrics.md)
- [文本算法介绍文档 (中文)](./docs/text_metrics.zh-CN.md)

### 图像算法

- [Image Evaluation Algorithm Document (English)](./docs/image_metrics.md)
- [图像数据评估使用文档 (中文)](./docs/image_metrics.zh-CN.md)

### 视频算法

- [Video Evaluation Algorithm Document (English)](./docs/video_metrics.md)
- [视频数据评估使用文档 (中文)](./docs/video_metrics.zh-CN.md)

## 数据评估论文总结(综述)

- [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
