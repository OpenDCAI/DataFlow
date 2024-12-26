# Video Data Processing

## 1. Pure Video Data Evaluation

### 👀 1.1 Dataset Preparation
Users can store the metadata of the dataset in the following JSON format:
```json
[
    {
        "flickr_id": 8536919744
    },
    {
        "flickr_id": 6408325533
    },
    {
        "flickr_id": 5319047612
    },
    {
        "flickr_id": 8724380666
    },
    {
        "flickr_id": 4744073127
    }
]
```

### 🌟 1.2 Writing the YAML Configuration File

Create a YAML configuration file for the dataset mentioned in section 1.1. The configuration under `data` specifies the dataset path and related information, and the configuration under `scorers` specifies the evaluation metrics you wish to use.

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2
dependencies: [video]
save_path: './example.jsonl'
data:
  video:
    meta_data_path: 'demos/video_process/video5data.json' # Path to meta data (mainly for image or video data)
    data_path: 'demos/video_process/videos/' # Path to dataset
    formatter: 'PureVideoFormatter' # Specify the data formatter

processors:
  VideoResolutionFilter:
    min_width: 160
    max_width: 7680
    min_height: 120
    max_height: 4320
    scorer_args:
      num_workers: 4
      batch_size: 1
  VideoMotionFilter:                              # Keep samples with video motion scores within a specific range.
    min_score: 0.25                                         # the minimum motion score to keep samples
    max_score: 10                                     # the maximum motion score to keep samples
    scorer_args:
      batch_size: 1
      num_workers: 4
      min_score: 0.25                                         # the minimum motion score to keep samples
      max_score: 10000.0                                      # the maximum motion score to keep samples
      sampling_fps: 2                                         # the samplig rate of frames_per_second to compute optical flow
      size: null                                              # resize frames along the smaller edge before computing optical flow, or a sequence like (h, w)
      max_size: null                                          # maximum allowed for the longer edge of resized frames
      relative: false                                         # whether to normalize the optical flow magnitude to [0, 1], relative to the frame's diagonal length
      any_or_all: any                                         # keep this sample when any/all videos meet the filter condition
processors:
  VideoResolutionFilter:
    min_width: 160
    max_width: 7680
    min_height: 120
    max_height: 4320
    scorer_args:
      num_workers: 4
      batch_size: 1
  VideoMotionFilter:                              # Keep samples with video motion scores within a specific range.
    min_score: 0.25                                         # the minimum motion score to keep samples
    max_score: 10                                     # the maximum motion score to keep samples
    scorer_args:
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

### 💪 1.3 Evaluating the Dataset
You can process (filter) the dataset with a single line of command
```bash
python process.py --config configs/process/video_process.yaml
```
Output is default stored in:
```
./scores.json
```
Output is in the following format:
```jsonl
{"flickr_id": 6408325533}
{"flickr_id": 5319047612}
{"flickr_id": 8724380666}
{"flickr_id": 4744073127}
```

## 2. Video-Text Data Evaluation

### 👀 2.1 Preparing the Dataset

Users can store the metadata of the dataset in the following JSON format:

```json
[
    {
        "videoID": "5-xGskbsBgI_000055_000065",
        "enCap": [
            "A young man is showing the polish, water, old soft cloth, and brush needed to polish shoes with.",
            "A person is setting up a table to get ready to shine boots.",
            "A man is giving instructions on how to polish with a brush and polisher.",
            "A guy is explaining the items you need to clean a pair of shoes.",
            "A young man is preparing to demonstrate how to shine shoes.",
            "A man picks up a can of shoe paste, a towel, and a brush from a table.",
            "A man is explaining what you need in order to shine shoes.",
            "A man instructs people on how to properly polish a pair of boots.",
            "A man is showing the products he uses to clean and shine shoes.",
            "A man discusses polishing an old pair of boots with an old t-shirt and a brush."
        ]
    },
    {
        "videoID": "uaoC__dKucA_000518_000528",
        "enCap": [
            "A woman explains how to sew a beanie as she sews a beanie in her hands.",
            "A lady is knitting a hat with straight knitting needles and she is explaining the process of what to do.",
            "A person is knitting with the help of both the hands and wool.",
            "A woman demonstrates how to crochet a knit cap using needles.",
            "A woman is showing how to do a knit stitch and a stocking knit stitch.",
            "A woman holds a knitted item and then gives knitting instructions.",
            "A woman demonstrates how to perform a stockinette stitch using needles and yarn.",
            "A woman is knitting a garment with gray yarn.",
            "A woman who is knitting a cap explains what the stitch count is.",
            "A woman demonstrates how to knit a hat using knitting needles."
        ]
    },
    {
        "videoID": "nJ6KPT1uwe8_000047_000057",
        "enCap": [
            "A woman is carving a pumpkin and gives instructions on how to do it.",
            "A woman is demonstrating how to clean a pumpkin properly.",
            "A person is using a very large knife to remove the core from a pumpkin and the insides of it.",
            "A person cleans the insides out of a pumpkin using some tools, while explaining the procedure.",
            "A person is explaining while another person is doing the actions of how to cut and scoop out a pumpkin.",
            "A person is showing how to clean out a pumpkin to make a jack-o'-lantern.",
            "A woman instructs and demonstrates how to remove the insides of a pumpkin.",
            "A voice is narrating while someone else is removing the insides of a pumpkin without breaking it.",
            "A person cuts a hole in the top of a pumpkin, then begins scooping out its insides.",
            "Someone is talking about and showing how to carve out the insides of a pumpkin."
        ]
    },
    {
        "videoID": "w-YJn8Je4GY_000062_000072",
        "enCap": [
            "A group of children play with different items on an outdoor cement slab.",
            "A group of people have a water balloon fight outside.",
            "A little boy throwing a water balloon at another boy outdoors.",
            "The children in green shirts play with small water balloons.",
            "Three kids are throwing water balloons at each other outside.",
            "A group of children play outside and one throws a water balloon.",
            "Three children are playing in a basketball court and a parent watches over them.",
            "Kids playing with water balloons, one of the kids attacks another with a water balloon.",
            "A group of boys and girls play with water balloons.",
            "A boy throws a rock at one of the kids outside."
        ]
    },
    {
        "videoID": "JSd8C0Ms-G0_000050_000060",
        "enCap": [
            "A red-haired girl is talking as she is rubbing her hands together in front of the camera.",
            "A young girl is rubbing lotion on her hands while she sits in her bedroom.",
            "A young lady rubs her hands together continuously while also talking.",
            "While sitting down on the floor, a little girl applies lotion to her hands.",
            "A teenage girl with red hair rubs lotion on her hands.",
            "A young girl sits down and fiddles with her arms while speaking to the camera.",
            "A woman is rubbing her hands together to spread lotion over them.",
            "An older girl rubs her hands together several times as if rubbing lotion in.",
            "A girl is rubbing her hands together and singing a song.",
            "A teenage girl with flaming red hair is explaining how to massage lotion into your hands."
        ]
    }
]
```

### 🌟 2.2 Writing the YAML Configuration File
Create a YAML configuration file for the dataset mentioned in section 2.1. The configuration under `data` specifies the dataset path and related information, and the configuration under `scorers` specifies the evaluation metrics you wish to use.

```yaml
model_cache_path: '../ckpt' # Path to cache models
num_workers: 2
dependencies: [video]
save_path: './example.jsonl'
data:
  video_caption:
    meta_data_path: 'demos/video_process/videocap5data.json' # Path to meta data (mainly for image or video data)
    data_path: 'demos/video_process/video-caption/'
    formatter: 'VideoCaptionFormatter'  # formatter for video-caption evaluation

processors:   
  EMScoreFilter:
    min_score: 0.3
    max_score: 1.0
    scorer_args:
      batch_size: 16
      num_workers: 4
  PACScoreFilter:
    min_score: 0.3
    max_score: 1.0
    scorer_args:
      batch_size: 16
      num_workers: 4
      model_path: ./models/clip_ViT-B-32.pth

```

### 💪 2.3 Evaluating the Dataset
You can process (filter) the dataset with a single line of command
```bash
python process.py --config configs/process/video_text_process.yaml
```
Output is default stored in:
```
./scores.json
```
Output is in the following format:
```jsonl
{"videoID": "nJ6KPT1uwe8_000047_000057", "enCap": ["A woman is carving a pumpkin and gives instructions how to do it", "A woman is demonstrating how to clean a pumpkin properly.", "A person is using a very large knife to remove the core from a pumpkin and the insides of it", "A person cleans the insides out of a pumpkin using some tools, while explaining the procedure", "A person is explaining while another person is doing the actions of how to cut and scoop out a pumpkin.", "A person showing how to clean out a pumpkin to make a jack o lantern.", "A woman instructs and demonstrates how to remove the insides of a pumpkin.", "A voice is narrating while someone else is removing the insides of a pumpkin without breaking it.", "A person cuts a hole in the top of a pumpkin, then begins scooping out its insides.", "Someone is talking about and showing how to carve out the insides of a pumpkin."]}
{"videoID": "JSd8C0Ms-G0_000050_000060", "enCap": ["A red haired girl is talking as she is rubbing her hands together in front of the camera.", "A young girl is rubbing lotion on her hands while she sits in her bedroom.", "A young lady rubs her hands together continuously while also talking.", "While sitting down on the floor, a little girl applies lotion to her hands.", "A teenage girl with red hair rubs lotion on her hands.", "A young girl sits down and fiddles with her arms while speaking to the camera.", "A woman is rubbing her hands together to spread lotion over them.", "An older girl rubs her hands together several times as if rubbing lotion in.", "A girl is rubbing her hands together and singing a song.", "A teenage girl with flaming red hair is explaining how to massage lotion into your hands."]}
```
