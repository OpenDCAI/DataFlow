
# 文本数据质量评估

本数据评估系统目前已整合了**20种不同类型的前沿文本数据评估方法**以及十余种生成文本评估方法。详见[评估算法文档](../../../docs/text_metrics.zh-CN.md)、[生成文本评估算法文档](../../../docs/gen_text_metrics.zh-CN.md)。在进行数据评估时，可通过`yaml`配置文件指定数据源、数据格式、打分器以及打分器配置信息。用户可通过更改配置文件的方式对不同的文本数据进行评估。


## 👀 配置文件

配置文件存放在`DataFlow/configs/eval`中，例如

```yaml
model_cache_path: '../ckpt' # 模型默认缓存路径
dependencies: [text] # 选择要加载的环境依赖
save_path: "./scores" # 输出分数存储路径

data:
  text:
    use_hf: False # 是否使用huggingface_dataset，如果使用则忽略下方本地数据地址，以下为Huggingface数据集参数设置；如果不使用则忽略。
    dataset_name: 'MBZUAI-LLM/SlimPajama-627B-DC'
    dataset_split: 'test'
    revision: 'refs/convert/parquet'
    name: 'default'
    data_path: 'demos/text_eval/fineweb_5_samples.json'  # 本地数据地址，支持json、jsonl、parquet格式
    formatter: "TextFormatter" # 数据加载器类型，使用TextFormatter即可

    keys: 'text' # 待评估的键名，对于sft数据，可指定为['instruction','input','output']
    
scorers: # 可从all_scorers.yaml中选择多个text打分器，将其配置信息放入即可
  PresidioScorer:
      language: 'en'
      device: 'cuda:0'
  QuratingScorer:
      model: 'princeton-nlp/QuRater-1.3B'
      tokens_field: 'input_ids'
      tokens: 512
      map_batch_size: 512
      num_workers: 1
      device_batch_size: 16
      device: 'cuda:0'
      labels:
        - writing_style
        - required_expertise
        - facts_and_trivia
        - educational_value
```

对于生成文本，配置文件需要指定待评估数据集和参考数据集文件以及键名。
```yaml
dependencies: [text] # 选择要加载的环境依赖
save_path: "./scores.json" # 输出分数存储路径
data:
  text:
    eval_data_path: "demos/text_eval/fineweb_5_samples.json" # 待评估数据路径
    ref_data_path: "demos/text_eval/alpaca_5_samples.json" # 参考数据路径
    ref_key: 'output' # 参考数据键名
    eval_key: 'text' # 待评估数据键名
    formatter: 'GenTextFormatter' # 数据加载器类型，使用GenTextFormatter即可

scorers:
  BleuScorer:
    n: 4 # Maximum value of N-gram
    eff: "average"  # Reference length selection method: "shortest", "average", "closest"
    special_reflen: null  # Set this value if a special reference sentence length is required

```

全部打分器配置保存在`DataFlow/configs/eval/all_scorers.yaml`中，生成文本评估打分器配置保存在`DataFlow/configs/eval/gen_text_scorers.yaml`中。使用时可以直接复制粘贴具体打分器配置信息。

## 🌟 数据集示例

本文本数据评估系统同时支持预训练数据和SFT数据格式。

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

## 💪 运行打分器
只需要一行代码便可运行：
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/text_scorer_example2.yaml
```
输出打分默认路径在
```
./scores.json
```
也可以自行在yaml的save_path指定

## 📌 输出示例
其中，`meta_scores`中保存对整个数据集层面的打分器得分，比如`VendiScore`。`item_scores`则保存数据集中每一条数据的单独得分。
```json
{
    "meta_scores": {},
    "item_scores": {
        "0": {
            "QuratingScore": {
                "QuratingWritingStyleScore": -0.3477,
                "QuratingRequiredExpertiseScore": -0.9062,
                "QuratingFactsAndTriviaScore": 1.789,
                "QuratingEducationalValueScore": 0.02051
            },
            "PresidioScore": {
                "Default": 6.0
            }
        },
        "1": {
            "QuratingScore": {
                "QuratingWritingStyleScore": -1.584,
                "QuratingRequiredExpertiseScore": -2.233,
                "QuratingFactsAndTriviaScore": -2.279,
                "QuratingEducationalValueScore": 1.518
            },
            "PresidioScore": {
                "Default": 4.0
            }
        },
        "2": {
            "QuratingScore": {
                "QuratingWritingStyleScore": 2.433,
                "QuratingRequiredExpertiseScore": 1.782,
                "QuratingFactsAndTriviaScore": 0.7237,
                "QuratingEducationalValueScore": 7.503
            },
            "PresidioScore": {
                "Default": 71.0
            }
        },
        "3": {
            "QuratingScore": {
                "QuratingWritingStyleScore": -2.444,
                "QuratingRequiredExpertiseScore": -0.1224,
                "QuratingFactsAndTriviaScore": 1.851,
                "QuratingEducationalValueScore": 4.234
            },
            "PresidioScore": {
                "Default": 16.0
            }
        },
        "4": {
            "QuratingScore": {
                "QuratingWritingStyleScore": -1.711,
                "QuratingRequiredExpertiseScore": -6.969,
                "QuratingFactsAndTriviaScore": -4.281,
                "QuratingEducationalValueScore": -6.125
            },
            "PresidioScore": {
                "Default": 2.0
            }
        }
    }
}
```
