# 图像数据质量评估

## 1. 纯图像数据质量评估
### 1.1 准备数据集（meta data）
用户可以将图像的id和文件名存储为如下标准json格式：
```json
[
    {
        "id": "0",
        "image": "10007903636.jpg"
    },
    {
        "id": "1",
        "image": "10089027076.jpg"
    }
]
```
<!-- 或者存为如下newline-delimited json格式
```json
{"id": "000114", "image": "000114.jpg"}
{"id": "000810", "image": "000810.jpg"}
``` -->

### 1.2 编写yaml配置文件
为1.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # 指定数据集的路径和相关信息
  image: # 要评估图像数据，因此在image下编写数据集配置信息
    meta_data_path: "../data/image_data.json" # 元数据的存放位置
    data_path: "../data/images" # 图像数据的存放位置
    image_key: 'image' # 元数据中图像路径（或图像名）对应的键
    id_key: 'id' # 元数据中id对应的键
    formatter: 'PureImageFormatter' # image数据固定使用PureImageFormatter

scorers: # 依次列出想使用的评估指标
  NiqeScorer:
      batch_size: 2
      device: "cpu"
```

### 1.3 评估数据集
编写好yaml配置文件后，调用 `calculate_score()` 即可对数据进行评估。
```python
from dataflow.utils.utils import calculate_score
calculate_score()
```
输出：
```
{
    'meta_scores': {}, 
    'item_scores': 
        {'0': 
            {
                'NiqeScorer': {'Default': 3.362590964504238} 
            }, 
        '1': 
            {
                'NiqeScorer': {'Default': 7.192364414148597}
            }
    }
}
```
## 2. 图像-文本数据评估
目前主要是图像-caption数据评估。对LLM的prompt稍作修改后即可用于图像SFT数据的评估。
### 2.1 准备数据集
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
<!-- 或者存为如下newline-delimited json格式
```json
{"id": "000114", "image": "000114.jpg", "caption": "an old man"}
{"id": "000810", "image": "000810.jpg", "caption": "blue sky"}
``` -->

### 2.2 编写yaml配置文件
为2.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # 指定数据集的路径和相关信息
  image_caption: # 要评估图像-caption数据，因此在image_caption下编写数据集配置信息
    meta_data_path: "../data/image_caption_data.json" # 元数据的存放位置
    data_path: "../data/images" # 图像数据的存放位置
    image_key: 'image' # 元数据中图像路径（或图像名）对应的键
    image_caption_key: 'caption' # 元数据中caption对应的键
    id_key: 'id' # 元数据中id对应的键
    formatter: 'ImageCaptionFormatter' # image数据固定使用ImageCaptionFormatter

scorers: # 依次列出想使用的评估指标
  ClipScorer:
      batch_size: 2
      device: "cuda"
  LongClipScorer:
      model_size: B # For larger models, use L
      batch_size: 2
      device: "cuda"
```

### 2.3 评估数据集
编写好yaml配置文件后，调用 `calculate_score()` 即可对数据进行评估。
```python
from dataflow.utils.utils import calculate_score
calculate_score()
```
输出：
```
{
    'meta_scores': {}, 
    'item_scores': 
    {
        '0': 
            {
                'ClipScorer': {'Default': 28.828125}, 
                'LongClipScorer': {'Default': 37.34375}
            }, 
        '1': 
            {
                'ClipScorer': {'Default': 33.4375}, 
                'LongClipScorer': {'Default': 35.3125}
            }
    }
}
```
<!-- ## 3 `calculate_score()` 函数背后的逻辑
```python
def calculate_score(save_path=None):
    from ..config import new_init_config
    from dataflow.utils.registry import FORMATTER_REGISTRY

    cfg = new_init_config()

    dataset_dict = {}
    for scorer_name, model_args in cfg.scorers.items(): # 依次加载yaml文件中指定的打分器
        if "num_workers" in cfg:
            model_args["num_workers"] = cfg.num_workers
        if "model_cache_path" in cfg:
            model_args["model_cache_dir"] = cfg.model_cache_path
        scorer = new_get_scorer(scorer_name, model_args)
        # 加载yaml文件中指定的数据集：
        if scorer.data_type not in dataset_dict:
            formatter = FORMATTER_REGISTRY.get(cfg['data'][scorer.data_type]['formatter'])(cfg['data'][scorer.data_type])
            dataset = formatter.load_dataset()
            dataset_dict[scorer.data_type] = dataset
        else:
            dataset = dataset_dict[scorer.data_type]

        # 使用打分器对数据集进行打分，并保存结果
        score = scorer(dataset)
        if isinstance(dataset, tuple):
            print(dataset[0].scores_list)
            print(dataset[0].meta_score)
        else:
            if len(dataset.meta_score.items()) > 0:
                print(dataset.meta_score)
        
    for _, dataset in dataset_dict.items():
        dataset.dump_scores(save_path)
``` -->