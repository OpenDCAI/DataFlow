[中文主页](./README.zh-CN.md)

# DataFlow-Eval

<p align="center">
  <img src="./static/images/Face.png">
</p>

DataFlow-Eval is a data evaluation system to evaluate data quality from multiple dimensions. We mainly support SOTA algorithms with in academic papers with strong theoretical support.

We now support text, image, video and multimodality data types.

## Table of Contents
- [DataFlow-Eval](#dataflow-eval)
  - [Table of Contents](#table-of-contents)
  - [Module and Modality Support](#module-and-modality-support)
  - [News](#news)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Text](#text)
    - [Image](#image)
    - [Video](#video)
  - [Data Evaluation Documentation](#data-evaluation-documentation)
    - [Text Documentation](#text-documentation)
    - [Image Documentation](#image-documentation)
    - [Video Documentation](#video-documentation)
  - [Data Evaluation Algorithms](#data-evaluation-algorithms)
    - [Text Algorithms](#text-algorithms)
    - [Image Algorithms](#image-algorithms)
    - [Video Algorithms](#video-algorithms)
  - [Awesome Data Evaluation](#awesome-data-evaluation)

## Module and Modality Support

| Module\Modality     | Text | Image | Video | Image-Text Pair | Video-Text Pair |
| ------------------- | ---- | ----- | ----- | --------------- | --------------- |
| **Data Evaluation** | ✅    | ✅     | ✅     | ✅               | ✅               |

## News

- [2024-10-15] We summarize data evaluation papers and codes in [Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
- [2024-10-15] Our first data-centric evaluation system is now open source.

## Installation


For environment setup, please using the following commands:

```
conda create -n datagym python=3.9
conda activate datagym
pip install -e .
```

If you want to evaluate each modality of data, please use the following commands:
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
For video data evaluation,
```bash
pip install -e .[video]
```
When evaluating video-caption data, please run the following command to install modified CLIP for EMScore:
```
pip install git+https://github.com/MOLYHECI/CLIP.git
```

</p>
</details>

All dependencies can be installed by:
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

Please refer to Data Evaluation Documentation for config file usage. Use the following command to run with the config file

```
python main.py --config [your config file]
```
<p align="center">
  <img src="./static/images/example_1.png">
</p>

## Quick Start
### Text
- [Text Evaluation Demo](./demos/text_eval/text_eval_example.ipynb)
- [文本评估示例](./demos/text_eval/text_eval_example.zh-CN.ipynb)
### Image
- [Image Evaluation Demo](./demos/image_eval/image_eval_example.ipynb)
- [图片评估示例](./demos/image_eval/image_eval_example.zh-CN.ipynb)
### Video
- [Video Evaluation Demo](./demos/video_eval/video_eval_example.ipynb)
- [视频评估示例](./demos/video_eval/video_eval_example.zh-CN.ipynb)

## Data Evaluation Documentation

For the usage of evaluation, please refer to the following documents.

### Text Documentation
- [Text Data Evaluation User Documentation (English)](./dataflow/Eval/Text/README.md)
- [文本数据评估使用文档 (中文)](./dataflow/Eval/Text/README.zh-CN.md)

### Image Documentation
- [Image Data Evaluation User Documentation (English)](./dataflow/Eval/image/README.md)
- [图像数据评估使用文档 (中文)](./dataflow/Eval/image/README.zh-CN.md)

### Video Documentation
- [Video Data Evaluation User Documentation (English)](./dataflow/Eval/video/README.md)
- [视频数据评估使用文档 (中文)](./dataflow/Eval/video/README.zh-CN.md)

## Data Evaluation Algorithms

We summarize the SOTA algorithms from academic papers for data evaluation.
### Text Algorithms
- [Text Evaluation Algorithm Document (English)](./docs/text_metrics.md)
- [文本算法介绍文档 (中文)](./docs/text_metrics.zh-CN.md)

### Image Algorithms
- [Image Evaluation Algorithm Document (English)](./docs/image_metrics.md)
- [图像数据评估使用文档 (中文)](./docs/image_metrics.zh-CN.md)

### Video Algorithms
- [Video Evaluation Algorithm Document (English)](./docs/video_metrics.md)
- [视频数据评估使用文档 (中文)](./docs/video_metrics.zh-CN.md)

## Awesome Data Evaluation
- [Awesome Data Evaluation](./Awesome_Data_Evaluation.md)
