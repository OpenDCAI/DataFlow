## Video Data Evaluation

### 1. Pure Video Data Evaluation

#### 1.1 Dataset Preparation
Users can store the metadata of their dataset in the following JSON format:
```json
[
    {
        "video": "test_video.mp4"
    },
    {
        "video": "test_video.mov"
    }
]
```

#### 1.2 Writing the YAML Configuration File

For the dataset from section 1.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information, while the `scorers` section defines the evaluation metrics to be used.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: './video.json' # Path to meta data (mainly for image or video data)
    data_path: './' # Path to dataset
    formatter: 'PureVideoFormatter' # Formatter for pure video evaluation

scorers:
  VideoMotionScorer:                              # Keep samples with video motion scores within a specific range.
      batch_size: 1
      num_workers: 4
      min_score: 0.25                             # Minimum motion score to keep samples
      max_score: 10000.0                          # Maximum motion score to keep samples
      sampling_fps: 2                             # Sampling rate of frames per second to compute optical flow
      size: null                                  # Resize frames along the smaller edge before computing optical flow, or a sequence like (h, w)
      max_size: null                              # Maximum allowed size for the longer edge of resized frames
      relative: false                             # Whether to normalize the optical flow magnitude to [0, 1], relative to the frame's diagonal length
      any_or_all: any                             # Keep this sample when any/all videos meet the filter condition
```

#### 1.3 Evaluating the Dataset
Once the YAML configuration file is ready, call the function `calculate_score()` to evaluate the data.

```python
from dataflow.utils.utils import calculate_score
calculate_score()
```
output:
```
{
    'meta_scores': {}, 
    'item_scores': 
    {
        '0': 
        {
            'VideoMotionScorer': {'Default': 0.6842129230499268}
        }, 
        '1': 
        {
            'VideoMotionScorer': {'Default': 8.972004890441895}
        }
    }
}
```


### 2. Video-Text Data Evaluation

#### 2.1 Dataset Preparation

Users can store the metadata of their dataset in the following JSON format:

```json
[
    {
        "video": "test_video.avi",
        "captions": [
            "A man is clipping paper.", 
            "A man is cutting paper."
        ]
    }
]
```

#### 2.2 Writing the YAML Configuration File

For the dataset from section 2.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information, while the `scorers` section defines the evaluation metrics to be used.

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: './video-caption.json' # Path to meta data (mainly for image or video data)
    data_path: './' # Path to dataset
    formatter: 'VideoCaptionFormatter' # Formatter for video-text evaluation

scorers:
  EMScorer:
    batch_size: 4
    num_workers: 4
```

#### 2.3 Evaluating the Dataset
Once the YAML configuration file is ready, call the function `calculate_score()` to evaluate the data.

```python
from dataflow.utils.utils import calculate_score
calculate_score()
```