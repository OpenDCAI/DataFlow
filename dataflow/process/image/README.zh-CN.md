# 图像数据处理

## 1. 纯图像数据质量处理
纯图像数据处理包括筛选和去重两部分：
* 筛选：依据打分器的打分以及其他启发式规则，滤除低质量图像。
* 去重：调用[imagededup](https://github.com/idealo/imagededup)对相似图像进行去重。
### 👀 1.1 准备数据集
用户可以将图像文件名存储为如下标准json格式：
```json
[
    {
        "image": "10007903636.jpg"
    },
    {
        "image": "10089027076.jpg"
    }
]
```

### 🌟 1.2 编写yaml配置文件
为1.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # 指定数据集的路径和相关信息
  image: # 要评估图像数据，因此在image下编写数据集配置信息
    meta_data_path: "demos/image_eval/image.json" # 元数据的存放位置
    data_path: "demos/image_eval/images" # 图像数据的存放位置
    image_key: 'image' # 元数据中图像路径（或图像名）对应的键
    formatter: 'PureImageFormatter' # image数据固定使用PureImageFormatter

processors: # 列出想要使用的过滤器和去重器
  # 过滤器
  # 指标位于[min_metric_name, max_metric_name]之外的图像将被去除，如果不想设置某个指标的过滤上限，则无需在yaml文件中设置max_metric_name；下限同理
  ImageResolutionFilter:
    min_width: 160
    max_width: 7680
    min_height: 120
    max_height: 4320
    batch_size: 2
  ImageAspectRatioFilter:
    min_ratio: 0.2
    max_ratio: 5.0
    batch_size: 2
  LiqeFilter:
    batch_size: 2
    device: "cuda"
    min_score: 3
    max_score: 5
  QalignFilter:
    batch_size: 2
    device: "cuda"
    min_score: 3
    max_score: 5
  # 去重器
  # 各去重器的threshold均应设置在0~64之间，阈值越低则过滤效果越强
  ImagePHashDeduplicator:
    threshold: 13  
  ImageDHashDeduplicator:
    threshold: 13 
  ImageWHashDeduplicator:
    threshold: 13 
  ImageAHashDeduplicator:
    threshold: 13 
```

### 💪 1.3 处理数据集
通过下面的一行代码处理数据集
```bash
python process.py --config configs/process/image_filter.yaml
```
输出将默认储存在下面的路径，也可以通过yaml中的save_path指定
```
./scores.json
```
算法将输出如下格式的数据:
```json
{"image": "10007903636.jpg"}
```

## 2. 图像-文本数据处理
依据打分器的分数对图像-文本数据进行过滤。
### 👀 2.1 准备数据集
用户可以将图像的id、文件名、图像对应的caption存储为如下标准json格式：

```json
[
    {
        "id": "0",
        "image": "cake.jpg",
        "caption": "a slice of chocolate cake on a white plate with a fork next to it"
    },
    {
        "id": "1",
        "image": "giraffe.jpg",
        "caption": "a giraffe reaching up to eat from a tree"
    }
]
```

### 🌟 2.2 编写yaml配置文件
为2.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # 指定数据集的路径和相关信息
  image_caption: # 要评估图像-caption数据，因此在image_caption下编写数据集配置信息
    meta_data_path: "demos/image_eval/image_text.json" # 元数据的存放位置
    data_path: "demos/image_eval/images" # 图像数据的存放位置
    image_key: 'image' # 元数据中图像路径（或图像名）对应的键
    image_caption_key: 'caption' # 元数据中caption对应的键
    id_key: 'id' # 元数据中id对应的键
    formatter: 'ImageCaptionFormatter' # image数据固定使用ImageCaptionFormatter

processors: # 依次列出想使用的过滤器
  ClipFilter:
    batch_size: 2
    device: "cuda"
    min_score: 30
  LongClipFilter:
    batch_size: 2
    device: "cuda"
    min_score: 25
    model_size: B
```

### 💪 2.3 处理数据集
通过下面的一行代码处理数据集
```bash
python process.py --config configs/process/image_text_filter.yaml
```
输出将默认储存在下面的路径，也可以通过yaml中的save_path指定
```
./scores.json
```
算法将输出如下格式的数据:
```json
{"image": "giraffe.jpg", "caption": "a giraffe reaching up to eat from a tree"}
```