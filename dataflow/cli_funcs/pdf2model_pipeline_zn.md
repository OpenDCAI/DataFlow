# DataFlow-pdf2model&LlaMA-Factory

## Quick Start

```
#环境配置
conda create -n dataflow python=3.10
conda activate dataflow
git clone https://github.com/OpenDCAI/DataFlow.git
cd DataFlow
pip install -e .[mineru]
pip install llamafactory[torch,metrics]
pip install open-dataflow[vllm]
#模型下载
#第一个两者都可以选
#第二个选all
mineru-models-download

#运行程序
cd ..
mkdir test
cd test

#初始化 
dataflow pdf2model init

#训练
dataflow pdf2model train

#与训练好的模型进行对话,也可以与本地训练好的模型对话
dataflow chat
```



## 第一步: 安装dataflow环境

```
#创建环境
conda create -n dataflow python=3.10

#激活环境
conda activate dataflow

#进入根目录
cd DataFlow

#下载mineru基础环境
pip install -e .[mineru]

#下载llamafactory环境
pip install llamafactory[torch,metrics]
pip install open-dataflow[vllm]
```



## 第二步: 创建新的dataflow工作文件夹

```
mkdir run_dataflow
cd run_dataflow
```



## 第三步: 设置数据集

将合适大小的数据集(数据文件为pdf格式)放到工作文件夹中



## 第四步: 初始化dataflow-pdf2model

```
#初始化 
#--cache 可以指定.cache目录的位置（可选）
#默认值为当前文件夹目录
dataflow pdf2model init
```

初始化完成后，项目目录变成：

```shell
项目根目录/
├── Pdf2QAPipeline.py  # pipeline执行文件
└── .cache/            # 缓存目录
    └── train_config.yaml  # llamafactory训练的默认配置文件
```



## 第五步: 一键微调

```
#--lf_yaml 可以指定训练所用llamafactory的yaml参数文件所在的路径(可选)
#默认值为.cache/train_config.yaml
dataflow pdf2model train
```

微调完成完成后，项目目录变成：

```
项目根目录/
├── Pdf2QAPipeline.py  # pipeline执行文件
└── .cache/            # 缓存目录
    ├── train_config.yaml  # llamafactory训练的默认配置文件
    ├── data/
    │   ├── dataset_info.json
    │   └── qa.json
    ├── gpu/
    │   ├── batch_cleaning_step_step1.json
    │   ├── batch_cleaning_step_step2.json
    │   ├── batch_cleaning_step_step3.json
    │   ├── batch_cleaning_step_step4.json
    │   └── pdf_list.jsonl
    ├── mineru/
    │   └── sample-1-7/auto/
    └── saves/
        └── qwen2.5_7b_sft_model/
```



## 第六步: 与微调好的模型对话

```
#--model 可以指定 对话模型的路径位置（可选）
#默认值为.cache/saves/qwen2.5_7b_sft_model
dataflow chat --model ./custom_model_path
```

