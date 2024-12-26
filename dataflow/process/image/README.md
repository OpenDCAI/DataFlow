# Image Data Processing

## 1. Pure Image Data Quality Processing
Pure image data processing includes two parts: screening and deduplication:
* Screening: Based on the scores given by scorers and other heuristic rules, low-quality images are filtered out.
* Deduplication: Invoke [imagededup](https://github.com/idealo/imagededup) to remove duplicate similar images.
### 👀 1.1 Preparing the Dataset
Users can store the image filename in the following standard JSON format:
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

### 🌟 1.2 Writing the YAML Configuration File
Write a YAML file in the following format for the dataset mentioned in section 1.1. The configurations under `data` specify the path and related information of the dataset, while those under `scorers` specify the evaluation metrics you wish to use.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # Specify the path and related information of the dataset
  image: # Since image data is to be evaluated, configure the dataset information under image
    meta_data_path: "demos/image_eval/image.json" # Location where metadata is stored
    data_path: "demos/image_eval/images" # Location where image data is stored
    image_key: 'image' # Key corresponding to the image path (or image name) in metadata
    formatter: 'PureImageFormatter' # Image data always uses PureImageFormatter

processors: # List the filters and deduplicators you want to use
  # Filters
  # Images with metrics outside the range [min_metric_name, max_metric_name] will be removed; if you do not want to set an upper limit for a metric's filter, then do not set max_metric_name in the YAML file; similarly for the lower limit
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
  # Deduplicators
  # The threshold for each deduplicator should be set between 0~64, the lower the threshold, the stronger the filtering effect
  ImagePHashDeduplicator:
    threshold: 13  
  ImageDHashDeduplicator:
    threshold: 13 
  ImageWHashDeduplicator:
    threshold: 13 
  ImageAHashDeduplicator:
    threshold: 13 
```

### 💪 1.3 Get Started
You can process (filter) the dataset with a single line of command
```bash
python process.py --config configs/process/image_filter.yaml
```
Output is default stored in:
```
./scores.json
```
Output is in the following format:
```json
{"image": "10007903636.jpg"}
```

## 2. Image-Text Data Processing
Filter image-text data based on the score given by scorers.
### 👀 2.1 Preparing the Dataset
Users can store the image filename and the caption corresponding to the image in the following standard JSON format:

```json
[
    {
        "image": "giraffe.jpg",
        "caption": "a deer eating grass"
    },
    {
        "image": "giraffe.jpg",
        "caption": "a giraffe reaching up to eat from a tree"
    }
]
```

### 🌟 2.2 Writing the YAML Configuration File
Write a YAML file in the following format for the dataset mentioned in section 2.1. The configurations under `data` specify the path and related information of the dataset, while those under `scorers` specify the evaluation metrics you wish to use.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data: # Specify the path and related information of the dataset
  image_caption: # Since image-caption data is to be evaluated, configure the dataset information under image_caption
    meta_data_path: "demos/image_eval/image_text.json" # Location where metadata is stored
    data_path: "demos/image_eval/images" # Location where image data is stored
    image_key: 'image' # Key corresponding to the image path (or image name) in metadata
    image_caption_key: 'caption' # Key corresponding to the caption in metadata
    id_key: 'id' # Key corresponding to the id in metadata
    formatter: 'ImageCaptionFormatter' # Image-caption data always uses ImageCaptionFormatter

processors: # List the filters you want to use in order
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

### 💪 2.3 Get Started
You can process (filter) the dataset with a single line of command
```bash
python process.py --config configs/process/image_text_filter.yaml
```
Output is default stored in:
```
./scores.json
```
Output is in the following format:
```json
{"image": "giraffe.jpg", "caption": "a giraffe reaching up to eat from a tree"}
```