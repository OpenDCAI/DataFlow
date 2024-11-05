# Text Data Quality Assessment

The current data evaluation system has integrated 20 different types of advanced text data evaluation methods. For details, see [Evaluation Algorithm Documentation](../../../docs/text_metrics_EN.md). During the data evaluation process, you can specify the data source, data format, scorer, and scorer configuration information through a `yaml` configuration file. Users can evaluate different text data by modifying the configuration file.


## Configuration File

The configuration file is located in `DataFlow/configs`, for example:

```yaml
model_cache_path: '../ckpt' # cache path for models

data:
  text:
    use_hf: False # Whether to use onlined Huggingface dataset, if used, ignore the local data path below
    dataset_name: 'yahma/alpaca-cleaned' # Huggingface dataset: dataset name
    dataset_split: 'train' # Huggingface dataset: dataset split
    name: 'default' # Huggingface dataset: subset name
    
    data_path: 'demos/text_eval/fineweb_5_samples.json'  # Local data path, supports json, jsonl, parquet formats
    formatter: "TextFormatter" # Data loader type

    keys: 'text' # Key name to be evaluated, for sft data, it can be specified as ['instruction','input','output']
    
scorers: # You can select multiple text scorers from all_scorers.yaml and put their configuration information here
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

## Dataset Example
This text data evaluation system supports both pre-training data and SFT data formats.

### Pre-training Dataset Example (excerpt from `Fineweb`):
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
To evaluate the above data format, you can specify `keys: Text`.

### SFT Dataset Example (excerpt from `alpaca-cleaned`):
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
To evaluate the above data format, you can specify `keys: ['instruction','input','output']`.

## Run scorers

```bash
cd path/to/DataFlow
python main.py --config /path/to/configfile
```
The `main.py` file is as follows. The path for saving the scoring results can be set through the `save_path` parameter.

```python
from dataflow.utils.utils import calculate_score
calculate_score(save_path='./scores.json')
```

## Output Sample
`meta_scores` stores the scores of the entire dataset level, such as `VendiScore`. `item_scores` stores the scores for each individual data in the dataset.
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
