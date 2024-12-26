# Image Data Quality Assessment

## 1. Pure Image Data Quality Assessment
### 👀 1.1 Prepare Dataset
Users can store the image id and file name in the following standard JSON format:
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
<!-- Or stored in the following newline-delimited JSON format
```json
{"id": "000114", "image": "000114.jpg"}
{"id": "000810", "image": "000810.jpg"}
``` -->

### 🌟 1.2 Write YAML Configuration File
For the dataset in section 1.1, write a YAML file in the following format, where the configuration under data specifies the path and related information of the dataset, and the configuration under scorers specifies the assessment metrics you want to use.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2
dependencies: [image]

data:
  image:
    meta_data_path: "demos/image_eval/image.json"
    data_path: "demos/image_eval/images"
    image_key: 'image'
    id_key: 'id'
    formatter: 'PureImageFormatter'

scorers:
  LiqeScorer:
      batch_size: 2
      device: "cuda"
  ArniqaScorer:
      batch_size: 2
      device: "cuda"
```

### 💪 1.3 Get Started 
You can assess the dataset with a single line of command
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/image_eval_example.yaml
```
Output is default stored in:
```
./scores.json
```
It should look like the following format:
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
## 2. Image-Text Data Assessment
Currently, it is mainly image-caption data assessment. After slightly modifying the prompt of LLM, it can be used for the assessment of image SFT data.
### 👀 2.1 Prepare Dataset
Users can store the image id, file name, and the corresponding caption of the image in the following standard JSON format:

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
<!-- Or stored in the following newline-delimited JSON format
```json
{"id": "000114", "image": "000114.jpg", "caption": "an old man"}
{"id": "000810", "image": "000810.jpg", "caption": "blue sky"}
``` -->

### 🌟 2.2 Write YAML Configuration File
For the dataset in section 2.1, write a YAML file in the following format, where the configuration under data specifies the path and related information of the dataset, and the configuration under scorers specifies the assessment metrics you want to use.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2
dependencies: [image]

data:
  image_caption:
    meta_data_path: "demos/image_eval/image_text.json"
    data_path: "demos/image_eval/images"
    image_key: 'image'
    image_caption_key: 'caption'
    id_key: 'id'
    formatter: 'ImageCaptionFormatter'

scorers:
  ClipScorer:
      batch_size: 2
      device: "cuda"
  LongClipScorer:
      model_size: B
      batch_size: 2
      device: "cuda"
```

### 💪 2.3 Assess the Dataset
After writing the YAML configuration file, call `calculate_score()` to assess the data.
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/image_text_eval_example.yaml
```
Output is default stored in:
```
./scores.json
```
It should look like the following format:
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
<!-- ## 3 Logic Behind `calculate_score()`
```python
def calculate_score(save_path=None):
    from ..config import new_init_config
    from dataflow.utils.registry import FORMATTER_REGISTRY

    cfg = new_init_config()

    dataset_dict = {}
    for scorer_name, model_args in cfg.scorers.items(): # Load the scorers specified in the yaml file one by one
        if "num_workers" in cfg:
            model_args["num_workers"] = cfg.num_workers
        if "model_cache_path" in cfg:
            model_args["model_cache_dir"] = cfg.model_cache_path
        scorer = new_get_scorer(scorer_name, model_args)
        # Load the dataset specified in the yaml file:
        if scorer.data_type not in dataset_dict:
            formatter = FORMATTER_REGISTRY.get(cfg['data'][scorer.data_type]['formatter'])(cfg['data'][scorer.data_type])
            dataset = formatter.load_dataset()
            dataset_dict[scorer.data_type] = dataset
        else:
            dataset = dataset_dict[scorer.data_type]

        # Use the scorer to score the dataset and save the results
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
