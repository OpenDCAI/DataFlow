# Text Data Processing

This data processing system integrates **over 60 processor methods, including deduplicators, filters, and refiners**. For more details, refer to the [Text Process Documentation](../../../docs/text_process.md). During data processing, you can specify the data source, format, processors, and processor configurations using a `yaml` configuration file. Users can modify the configuration file to process different types of text data.

## 👀 Configuration File

Configuration files are stored in `DataFlow/configs/process`. Below is a simple example (`DataFlow/configs/process/text_process_example.yaml`) that includes one refiner, one deduplicator, and one filter.

```yaml
model_cache_path: '../ckpt' # Path to model cache
dependencies: [text]
save_path: './processed.jsonl' # Path to save processed data

data:
  text:
    use_hf: False # Whether to load a Huggingface dataset. If true, the local path below is ignored.
    dataset_name: 'yahma/alpaca-cleaned'
    dataset_split: 'train'
    name: 'default'
    revision: null
    data_path: 'demos/text_eval/example.json'  # Local data path, supports json, jsonl, parquet formats
    formatter: "TextFormatter" # Data loader type; use TextFormatter

    keys: 'text' # Keys to process. For SFT data, specify ['instruction','input','output']

processors: # Data processors
  RemoveExtraSpacesRefiner: {}
  CCNetDeduplicator: 
    bit_length: 64 
  NgramFilter:
    min_score: 0.99
    max_score: 1.0
    scorer_args:
      ngrams: 5
```
The full configuration for scorers is stored in `DataFlow/configs/process/text_process.yaml`. You can directly copy specific scorer configurations into the process field for sequential processing.

## 🌟 Dataset Examples
This text data processing system supports both pre-training and SFT data formats.

### Pre-training Dataset Example (excerpt from Fineweb):
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
To evaluate the above data format, specify keys: text.

### SFT Dataset Example (excerpt from alpaca-cleaned):
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
To evaluate the above data format, specify `keys: ['instruction','input','output']`.

## 💪 Running the Processor
You can process (filter) the dataset with a single line of command
```bash
cd path/to/DataFlow
python process.py --config configs/process/text_process_example.yaml
```
Output is default stored in:
```
./scores.json
```

## 📌 Example Execution
The sample dataset (`demos/text_process/fineweb_5_samples.json`) contains 1 pair of duplicate data, 1 piece of low-quality data with n-gram repetitions. Additionally, most of the data includes extra spaces.
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




