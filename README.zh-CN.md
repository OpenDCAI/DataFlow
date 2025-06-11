[English Readme](./README.md)

# DataFlow

<div align="center">
  <img src="./static/images/Face.png">

[![](https://img.shields.io/github/license/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/Open-DataFlow/DataFlow?style=social)](https://github.com/Open-DataFlow/DataFlow)
[![](https://img.shields.io/github/issues-raw/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/issues)
[![](https://img.shields.io/github/last-commit/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/commits/main/)
[![](https://img.shields.io/github/contributors/Open-DataFlow/DataFlow)](https://github.com/Open-DataFlow/DataFlow/graphs/contributors)

简体中文 | [English](./README.md)

[功能特点](#功能特点) • [快速开始](#本地运行) • [使用文档](https://docs.easy-dataset.com/) • [贡献](#贡献) • [许可证](#许可证)


</div>

DataFlow-Eval 是一个数据质量和处理的评估系统，可以从多个维度评估数据质量并筛选高质量数据。我们主要支持具有强大理论支持的学术论文中的最新算法。

我们目前支持文本、图像、视频和多模态数据类型。

## Table of Contents
- [DataFlow](#dataflow)
  - [Table of Contents](#table-of-contents)
  - [模块和模态支持](#模块和模态支持)
  - [新闻](#新闻)
  - [安装](#安装)
  - [快速开始](#快速开始)
    - [快速评估:](#快速评估)
    - [快速处理:](#快速处理)
  - [Jupyter Notebook Demo](#jupyter-notebook-demo)
    - [文本示例](#文本示例)
    - [图像示例](#图像示例)
    - [视频示例](#视频示例)
  - [数据评估\&处理文档](#数据评估处理文档)
    - [文本文档](#文本文档)
    - [图像文档](#图像文档)
    - [视频文档](#视频文档)
  - [数据评估\&处理算法](#数据评估处理算法)
  - [数据评估论文总结(综述)](#数据评估论文总结综述)

## 模块和模态支持

| 模块\模态    | 文本  | 图像  | 视频  | 图像-文本对 | 视频-文本对 |
| -------- | --- | --- | --- | ------ | ------ |
| **数据评估** | ✅   | ✅   | ✅   | ✅      | ✅      |


## 新闻

- [2024-12-26] 🎉 我们的评估与数据处理系统开源了
- [2024-10-14] 🎉 我们在 [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)总结了数据评估相关论文
- [2024-10-14] 🎉 我们的数据评估系统开源了

## 安装

您可以用如下命令配置conda环境
```
conda create -n dataflow python=3.10

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
当评估video-caption数据时, 请运行下列代码下载EMScore定制的CLIP:
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
  
## 快速开始
### 快速评估:
```
cd path/to/DataFlow
python eval.py --config configs/eval/text_scorer_example1.yaml
python eval.py --config configs/eval/image_eval_example.yaml
python eval.py --config configs/eval/video_scorer.yaml
```
### 快速处理:
```
cd path/to/DataFlow
python process.py --config configs/process/text_process_example.yaml
python process.py --config configs/process/image_filter.yaml
python process.py --config configs/process/video_process.yaml
```
config中的yaml都可以直接跑

## Jupyter Notebook Demo
### 文本示例
- [Text Evaluation Demo](./demos/text_eval/text_eval_example.ipynb)
- [文本评估示例](./demos/text_eval/text_eval_example.zh-CN.ipynb)
- [Text Process Demo](./demos/text_process/text_process_example.ipynb)
- [文本处理示例](./demos/text_process/text_process_example.zh-CN.ipynb)

### 图像示例
- [Image Evaluation Demo](./demos/image_eval/image_eval_example.ipynb)
- [图片评估示例](./demos/image_eval/image_eval_example.zh-CN.ipynb)
- [Image Process Demo](./demos/image_process/image_process_example.ipynb)
- [图片处理示例](./demos/image_process/image_process_example.zh-CN.ipynb)

### 视频示例
- [Video Evaluation Demo](./demos/video_eval/video_eval_example.ipynb)
- [视频评估示例](./demos/video_eval/video_eval_example.zh-CN.ipynb)
- [Video Process Demo](./demos/video_process/video_process_example.ipynb)
- [视频处理示例](./demos/video_process/video_process_example.zh-CN.ipynb)

使用CLIPScore打分器的评估示例:
<p align="center">
  <img src="./static/images/example_1.png">
</p>

## 数据评估&处理文档

请参照下面的文档了解不同模态的数据评估👇

### 文本文档

- [Text Data Evaluation User Documentation (English)](./dataflow/Eval/Text/README.md)
- [文本数据评估使用文档 (中文)](./dataflow/Eval/Text/README.zh-CN.md)
- [Text Data Process User Documentation (English)](./dataflow/process/text/README.md)
- [文本数据处理使用文档 (中文)](./dataflow/process/text/README.zh-CN.md)

### 图像文档

- [Image Data Evaluation User Documentation (English)](./dataflow/Eval/image/README.md)
- [图像数据评估使用文档 (中文)](./dataflow/Eval/image/README.zh-CN.md)
- [Image Data Process User Documentation (English)](./dataflow/process/image/README.md)
- [图像数据处理使用文档 (中文)](./dataflow/process/image/README.zh-CN.md)

### 视频文档

- [Video Data Evaluation User Documentation (English)](./dataflow/Eval/video/README.md)
- [视频数据评估使用文档 (中文)](./dataflow/Eval/video/README.zh-CN.md)
- [Video Data Process User Documentation (English)](./dataflow/process/video/README.md)
- [视频数据处理使用文档 (中文)](./dataflow/process/video/README.zh-CN.md)

## 数据评估&处理算法

[Dataflow 文档](https://open-dataflow.github.io/DataFlow-Eval-Process/)

## 数据评估论文总结(综述)

- [👋 Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
