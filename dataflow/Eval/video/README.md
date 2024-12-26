# Video Data Evaluation

## 1. Pure Video Data Evaluation

### 👀 1.1 Dataset Preparation
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

### 🌟 1.2 Writing the YAML Configuration File

For the dataset from section 1.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information, while the `scorers` section defines the evaluation metrics to be used.
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: 'demos/video_eval/video.json' # Path to meta data (mainly for image or video data)
    data_path: 'demos/video_eval/' # Path to dataset
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

### 💪 1.3 Get Started
You can assess the dataset with a single line of command
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/video_scorer.yaml
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
            'VideoMotionScorer': {'Default': 0.6842129230499268}
        }, 
        '1': 
        {
            'VideoMotionScorer': {'Default': 8.972004890441895}
        }
    }
}
```


## 2. Video-Text Data Evaluation

### 👀 2.1 Dataset Preparation

Users can store the metadata of their dataset in the following JSON format:

```json
[
    {
        "video": "test_video.avi",
        "enCap": [
            "A man is clipping paper.", 
            "A man is cutting paper."
        ]
    }
]
```

### 🌟 2.2 Writing the YAML Configuration File

For the dataset from section 2.1, write a YAML file in the following format. The `data` section specifies the dataset path and related information, while the `scorers` section defines the evaluation metrics to be used.

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: 'demos/video_eval/video-caption.json' # Path to meta data (mainly for image or video data)
    data_path: 'demos/video_eval/' # Path to dataset
    formatter: 'VideoCaptionFormatter' # Formatter for video-text evaluation

scorers:
  EMScorer:
    batch_size: 4
    num_workers: 4
```

### 💪 2.3 Evaluating the Dataset
You can assess the dataset with a single line of command
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/video_text_scorer.yaml
```
Output is default stored in:
```
./scores.json
```
It should look like the following format:
```
{
    "meta_scores": {},
    "item_scores": {
        "0": {
            "EMScorer": {
                "EMScore(X,X*)": {
                    "figr_P": 0.9121,
                    "figr_R": 0.9121,
                    "figr_F": 0.9121,
                    "cogr": 0.934,
                    "full_P": 0.9231,
                    "full_R": 0.9231,
                    "full_F": 0.9231
                },
                "EMScore(X,V)": {
                    "figr_P": 0.228,
                    "figr_R": 0.2537,
                    "figr_F": 0.2402,
                    "cogr": 0.2598,
                    "full_P": 0.2439,
                    "full_R": 0.2568,
                    "full_F": 0.25
                },
                "EMScore(X,V,X*)": {
                    "figr_P": 0.5701,
                    "figr_R": 0.5829,
                    "figr_F": 0.5762,
                    "cogr": 0.5969,
                    "full_P": 0.5835,
                    "full_R": 0.5899,
                    "full_F": 0.5866
                }
            }
        }
    }
}
```
