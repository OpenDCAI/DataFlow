## 视频数据评估

### 1. 纯视频数据评估

#### 1.1 数据集准备
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


#### 2.编写yaml配置文件

为1.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。
```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: './video.json' # Path to meta data (mainly for image or video data)
    data_path: './' # Path to dataset
    formatter: 'PureVideoFormatter' # formatter for pure video evaluation

scorers:
  VideoMotionScorer:                              # Keep samples with video motion scores within a specific range.
      batch_size: 1
      num_workers: 4
      min_score: 0.25                                         # the minimum motion score to keep samples
      max_score: 10000.0                                      # the maximum motion score to keep samples
      sampling_fps: 2                                         # the samplig rate of frames_per_second to compute optical flow
      size: null                                              # resize frames along the smaller edge before computing optical flow, or a sequence like (h, w)
      max_size: null                                          # maximum allowed for the longer edge of resized frames
      relative: false                                         # whether to normalize the optical flow magnitude to [0, 1], relative to the frame's diagonal length
      any_or_all: any                                         # keep this sample when any/all videos meet the filter condition
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

#### 1.3 评估数据集
编写好yaml配置文件后，调用```calculate_score()```即可对数据进行评估。

```
from dataflow.utils.utils import calculate_score
calculate_score()
```

### 2. 视频-文本数据评估

#### 2.1 准备数据集

用户可以将数据集的元数据存储成如下json格式:

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

#### 2.2 编写yaml配置文件
为2.1节的数据集编写如下格式的yaml文件，其中data下的配置用于指定数据集的路径和相关信息，scorers下的配置用于指定您想使用的评估指标。

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2

data:
  video:
    meta_data_path: './video-caption.json' # Path to meta data (mainly for image or video data)
    data_path: './' # Path to dataset
    formatter: 'VideoCaptionFormatter' # formatter for pure video evaluation

scorers:
  EMScorer:
    batch_size: 4
    num_workers: 4

```

#### 2.3 评估数据集
编写好yaml配置文件后，调用```calculate_score()```即可对数据进行评估。

```
from dataflow.utils.utils import calculate_score
calculate_score()
```

