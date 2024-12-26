# 视频数据评估

## 1. 纯视频数据评估

### 👀 1.1 数据集准备
用户可以将数据集的元数据存储成如下json格式:
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


### 🌟 1.2 编写yaml配置文件

为1.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
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
输出:
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

### 💪 1.3 评估数据集
可以用一行代码完成评估:
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/video_scorer.yaml
```
输出被保存在:
```
./scores.json
```
输出格式如下:

## 2. 视频-文本数据评估

### 👀 2.1 准备数据集

用户可以将数据集的元数据存储成如下json格式:

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

### 🌟 2.2 编写yaml配置文件
为2.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。

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

### 💪 2.3 评估数据集
可以用一行代码完成评估:
```bash
cd path/to/DataFlow
python eval.py --config configs/eval/video_text_scorer.yaml
```
输出被保存在:
```
./scores.json
```
输出格式如下:
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
