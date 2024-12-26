# Video Data Processing
The processing of video data primarily relies on dataset filtering methods based on evaluation scores.

## Pure Video Processing
### Method Categories
| Category Description | Metric List |
|--- |--- |
| Based on Video Statistics | Motion Score |
| Based on Pre-trained Models | FastVQAScorer, FasterVQAScorer, DOVERScorer |

### Method Overview
| Name | Filtering Metric | Filtering Dimension | Scorer Introduction | Score Range |
| ---- | ---- | ---- | ---- | ---- |
| VideoMotionFilter | Motion Score | Statistics |  Calculates the magnitude of optical flow vectors between frames as the score |  |
| [FastVQAFilter](https://arxiv.org/abs/2207.02595v1) | Pre-trained model Scoring | Model | Scorer based on Video Swin Transformer, incorporating the Fragment Sampling module, which improves accuracy and speed | [0,1] |
| [FasterVQAFilter](https://arxiv.org/abs/2210.05357) | Pre-trained model Scoring | Model |An optimized version of FastVQAScorer, with improvements to the Fragment Sampling module, achieving significant speed enhancements | [0,1] |
| [DOVERFilter](https://arxiv.org/abs/2211.04894) | Pre-trained model scoring | Model |Based on FastVQAScorer, it provides scores from both technical and aesthetic perspectives	 |  |

## Video-Text Processing
| Category Description | Metric List |
|--- |--- |
| Based on pre-trained vision-language models | EMScore, PAC-S |

| Name | Filtering Metric | Filtering Dimension | Scorer Introduction | Score Range |
| ---- | ---- | ---- | ---- | ---- |
| [EMScorer](https://arxiv.org/abs/2111.08919) | Video-Text Similarity Scoring | Model | A video-text scorer based on CLIP, supporting both with-reference and no-reference scoring. | [0,1] |
| [PACScorer](https://arxiv.org/abs/2303.12112) | ideo-Text Similarity Scoring | Model | A video-text scorer based on CLIP, with tuned CLIP Encoder on top of EMScore | [0,1] |