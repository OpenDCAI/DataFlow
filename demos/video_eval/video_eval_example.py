### Video Data Evaluation Methods

# The evaluation methods for video data mainly fall into the following categories:
# - Motion Score
# - DL-based Score
# - CLIP-based Score

# For detailed configuration examples, please refer to the file `configs/video_score.yaml`. Below is a simple format for a YAML configuration file (`./video_eval.yaml`):

# ```yaml
# model_cache_path: '../ckpt' # Path to cache models
# num_workers: 2

# data:
#   video:
#     meta_data_path: './video.json' # Path to meta data (mainly for image or video data)
#     data_path: './' # Path to dataset
#     formatter: 'PureVideoFormatter' # Formatter for pure video evaluation
# ```

# The `data` section specifies the paths and relevant configurations for the data files or folders.

# ```yaml
# scorers:
#   VideoMotionScorer:                              # Retain samples with video motion scores within a specific range.
#       batch_size: 1
#       num_workers: 4
#       min_score: 0.25                             # Minimum motion score to retain samples
#       max_score: 10000.0                          # Maximum motion score to retain samples
#       sampling_fps: 2                             # Sampling rate of frames per second for computing optical flow
#       size: null                                  # Resize frames along the smaller edge before computing optical flow, or a sequence like (h, w)
#       max_size: null                              # Maximum allowed size for the longer edge of resized frames
#       relative: false                             # Whether to normalize the optical flow magnitude to [0, 1], relative to the frame's diagonal length
#       any_or_all: any                             # Retain the sample when any/all videos meet the filter condition
# ```

# The `scorers` section defines the parameter configurations for the scorers in use.

import sys
import os

dataflow_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 
sys.path.insert(0, dataflow_path)

import dataflow
from dataflow.utils import calculate_score

calculate_score()