
# 文本数据处理

本数据处理系统目前已经整合了包括去重、过滤、改写在内的六十余种处理器方法，详见[数据处理器文档](../../../docs/text_process.zh-CN.md)。在进行数据处理时，可通过`yaml`配置文件指定数据源、数据格式、处理器以及处理器配置信息。用户可通过更改配置文件的方式对不同的文本数据进行处理。


## 👀 配置文件

配置文件存放在`DataFlow/configs/process`中，例如以下简单的配置文件（`DataFlow/configs/process/text_process_example.yaml`）包含了一个改写器、一个去重器和一个过滤器。

```yaml

model_cache_path: '../ckpt' # 模型缓存路径
dependencies: [text]
save_path: './processed.jsonl' # 处理后数据的存储地址

data:
  text:
    use_hf: False # 是否加载Huggingface的数据集，如果加载，则忽略下方本地地址
    dataset_name: 'yahma/alpaca-cleaned'
    dataset_split: 'train'  
    name: 'default' 
    revision: null
    data_path: 'demos/text_process/fineweb_5_samples.json'  # 本地数据地址，支持json、jsonl、parquet等格式
    formatter: "TextFormatter" # 数据加载器类型，使用TextFormatter即可

    keys: 'text' # 需要处理的键名，对于sft数据，指定为['instruction','input','output']

processors: # 数据处理器
  RemoveExtraSpacesRefiner: {}
  CCNetDeduplicator: 
    bit_length: 64 
  NgramFilter:
    min_score: 0.99
    max_score: 1.0
    scorer_args:
      ngrams: 5
```

全部打分器配置保存在`DataFlow/configs/process/text_process.yaml`中。使用时可以直接复制粘贴具体打分器配置信息到`process`字段中按顺序处理。

## 🌟 数据集示例

本文本数据处理系统同时支持预训练数据和SFT数据格式。

### 预训练数据集示例（摘自`Fineweb`）：
```json
[
    {
        "text": "On Tuesday, NASCAR announced the release of \u201cNASCAR Classic Races, Volume 1,\u201d available on iTunes.",
        "id": "<urn:uuid:5189a256-bd76-489b-948e-9300a6f3f9da>"
    },
    {
        "text": "Tiger, GA Homeowners Insurance\nGet cheap home insurance in Tiger, GA within minutes. ",
        "id": "<urn:uuid:b49eaf47-48ed-4ff1-9121-f9e36247831f>"
    }
]
```
若要对上述数据格式进行评估，可指定`keys: text`

### SFT数据集示例（摘自`alpaca-cleaned`）
```json
[
    {
        "instruction": "Rearrange the following sentence to make the sentence more interesting.",
        "input": "She left the party early",
        "output": "Early, she left the party."
    },
    {
        "instruction": "Let \n f(x) = {[ -x - 3 if x \u2264 1,; x/2 + 1 if x > 1. ].\nFind the sum of all values of x such that f(x) = 0.",
        "input": "",
        "output": "We solve the equation f(x) = 0 on the domains x \u2264 1 and x > 1.\n\nIf x \u2264 1, then f(x) = -x - 3, so we want to solve -x - 3 = 0. The solution is x = -3, which satisfies x \u2264 1.\n\nIf x > 1, then f(x) = x/2 + 1, so we want to solve x/2 + 1 = 0. The solution is x = -2, but this value does not satisfy x > 1.\n\nTherefore, the only solution is x = -3."
    }
]
```
若要对上述数据格式进行评估，可指定`keys: ['instruction','input','output']`

## 💪 运行处理器
通过下面的一行代码处理数据集
```bash
cd path/to/DataFlow
python process.py --config configs/process/text_process_example.yaml
```
输出将默认储存在下面的路径，也可以通过yaml中的save_path指定
```
./scores.json
```

## 📌 运行示例

本示例数据集（`demos/text_process/fineweb_5_samples.json`）中共含有包含1对重复数据，一个ngram重复低质量数据。同时大部分数据含有多余空格。

```bash
RemoveExtraSpacesRefiner {'num_workers': 1, 'model_cache_dir': '../ckpt'}
Generating train split: 5 examples [00:00, 154.94 examples/s]
Implementing RemoveExtraSpacesRefiner: 100%|██████████| 5/5 [00:00<00:00, 4314.24it/s]
Implemented RemoveExtraSpacesRefiner. 4 data refined.
CCNetDeduplicator {'bit_length': 64, 'num_workers': 1, 'model_cache_dir': '../ckpt'}
Module dataflow.process.text.refiners has no attribute CCNetDeduplicator
Module dataflow.process.text.filters has no attribute CCNetDeduplicator
Implementing CCNetDeduplicator: 100%|██████████| 5/5 [00:00<00:00, 81601.25it/s]
Implemented CCNetDeduplicator. Data Number: 5 -> 4
NgramFilter {'min_score': 0.99, 'max_score': 1.0, 'scorer_args': {'ngrams': 5}, 'num_workers': 1, 'model_cache_dir': '../ckpt'}
Module dataflow.process.text.refiners has no attribute NgramFilter
Evaluating NgramScore: 100%|██████████| 4/4 [00:00<00:00, 260.35it/s]
Implemented NgramFilter. Data Number: 4 -> 3
Data saved to ./processed.jsonl
```
