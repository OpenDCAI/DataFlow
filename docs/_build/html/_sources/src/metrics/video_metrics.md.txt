# Video Data Evaluation Metrics

## Pure Video Evaluation Metrics
### Metric Categories
| Category Description | Metric List | 
|--- |--- |
| Based on Video Statistics | Motion Score | 
| Based on Pre-trained Models | FastVQAScorer, FasterVQAScorer, DOVERScorer |

### Metric Descriptions
| Name | Evaluation Metric | Dimension | Description | Value Range |  
| ---- | ---- | ---- | ---- | ---- | 
| VideoMotionScorer | Motion Score | Statistical | Calculates the magnitude of optical flow vectors between frames as the score |  | 
| [FastVQAScorer](https://arxiv.org/abs/2207.02595v1) | Pre-trained Model Scoring | Model |Scorer based on Video Swin Transformer, incorporating the Fragment Sampling module, which improves accuracy and speed | [0,1] | 
| [FasterVQAScorer](https://arxiv.org/abs/2210.05357) | Pre-trained Model Scoring | Model |An optimized version of FastVQAScorer, with improvements to the Fragment Sampling module, achieving significant speed enhancements | [0,1] | 
| [DOVERScorer](https://arxiv.org/abs/2211.04894) | Pre-trained Model Scoring | Model | Based on FastVQAScorer, it provides scores from both technical and aesthetic perspectives	 |  |

### Reference Values
To better provide data quality references, we evaluated the KoNViD-1k dataset using the above metrics, and the distribution of the evaluation values is as follows:
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Metric</th>
    <th class="tg-0pky">Name</th>
    <th class="tg-0pky">Mean</th>
    <th class="tg-0pky">Variance</th>
    <th class="tg-0pky">Max</th>
    <th class="tg-0pky">Min</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Motion Score</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Calculates the magnitude of the optical flow vectors between video frames as the score. The stronger the motion in the video, the greater the frame-to-frame changes, and the higher the score</td>
    <td class="tg-0pky">6.2745</td>
    <td class="tg-0pky">19.28</td>
    <td class="tg-0pky">25.23</td>
    <td class="tg-0pky">0.001623</td>
  </tr>
  <tr>
    <td class="tg-0pky" >FastVQA</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">The score obtained using the FastVQAScorer module. The better the video quality, the higher the score</td>
    <td class="tg-0pky">0.4987</td>
    <td class="tg-0pky">0.04554</td>
    <td class="tg-0pky">0.9258</td>
    <td class="tg-0pky">0.007619</td>
  </tr>
  <tr>
    <td class="tg-0pky">FasterVQA</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">The score obtained using the FasterVQAScorer module. The better the video quality, the higher the score</td>
    <td class="tg-0pky">0.5134</td>
    <td class="tg-0pky">0.04558</td>
    <td class="tg-0pky">0.9066</td>
    <td class="tg-0pky">0.03686</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">DOVER</td>
    <td class="tg-0pky">technical</td>
    <td class="tg-0pky">One of the scores obtained using the DOVERScorer module, the better the technical quality of the video, the higher the score</td>
    <td class="tg-0pky">-0.1107</td>
    <td class="tg-0pky">0.001755</td>
    <td class="tg-0pky">-0.006550</td>
    <td class="tg-0pky">-0.3175</td>
  </tr>
  <tr>
    <td class="tg-0pky">aesthetic</td>
    <td class="tg-0pky">One of the scores obtained using the DOVERScorer module, the better the aesthetic quality of the video, the higher the score</td>
    <td class="tg-0pky">-0.008419</td>
    <td class="tg-0pky">0.004569</td>
    <td class="tg-0pky">0.1869</td>
    <td class="tg-0pky">-0.2629</td>
  </tr>
</tbody></table>

## Video-Text Evaluation Metrics
| Category Description | Metric List | 
|--- |--- |
| Based on Pre-trained Vision-Language Models | EMScore, PAC-S | 

| Name | Evaluation Metric | Dimension | Description | Value Range |
| ---- | ---- | ---- | ---- | ---- |
| [EMScorer](https://arxiv.org/abs/2111.08919) | Video-Text Similarity Scoring | Model | A video-text scorer based on CLIP, supporting both with-reference and no-reference scoring. | [0,1] |
| [PACScorer](https://arxiv.org/abs/2303.12112) | Video-Text Similarity Scoring | Model | A video-text scorer based on CLIP, with tuned CLIP Encoder on top of EMScore| [0,1] |

### Reference Values
To provide better data quality reference, we evaluated the VATEX dataset ([link](https://huggingface.co/datasets/lmms-lab/VATEX)) using the above metrics, and the distribution of the metric values is as follows:

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Metric</th>
    <th class="tg-0pky">Name</th>
    <th class="tg-0pky">Mean</th>
    <th class="tg-0pky">Variance</th>
    <th class="tg-0pky">Max</th>
    <th class="tg-0pky">Min</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">EMScorer</td>
    <td class="tg-0pky">figr_F</td>
    <td class="tg-0pky">The score obtained using the EMScorer module. The higher the similarity between the video and text at a fine-grained level, the higher the score	</td>
    <td class="tg-0pky">0.2712</td>
    <td class="tg-0pky">0.0003667</td>
    <td class="tg-0pky">0.3461</td>
    <td class="tg-0pky">0.1987</td>
  </tr>
  <tr>
    <td class="tg-0pky">cogr</td>
    <td class="tg-0pky">The score obtained using the EMScorer module. The higher the similarity between the video and text at a coarse-grained level, the higher the score	</td>
    <td class="tg-0pky">0.3106</td>
    <td class="tg-0pky">0.0009184</td>
    <td class="tg-0pky">0.4144</td>
    <td class="tg-0pky">0.18</td>
  </tr>
    <tr>
    <td class="tg-0pky">full_F</td>
    <td class="tg-0pky">The score obtained using the EMScorer module, which is the arithmetic mean of the above two scores</td>
    <td class="tg-0pky">0.2909</td>
    <td class="tg-0pky">0.0005776</td>
    <td class="tg-0pky">0.3712</td>
    <td class="tg-0pky">0.3807</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">PACScorer</td>
    <td class="tg-0pky">figr_F</td>
    <td class="tg-0pky">The score obtained using the PACScorer module. The higher the similarity between the video and text at a fine-grained level, the higher the score</td>
    <td class="tg-0pky">0.36553</td>
    <td class="tg-0pky">0.0004902</td>
    <td class="tg-0pky">0.4456</td>
    <td class="tg-0pky">0.2778</td>
  </tr>
  <tr>
    <td class="tg-0pky">cogr</td>
    <td class="tg-0pky">The score obtained using the PACScorer module. The higher the similarity between the video and text at a coarse-grained level, the higher the score	</td>
    <td class="tg-0pky">0.4160</td>
    <td class="tg-0pky">0.001021</td>
    <td class="tg-0pky">0.5222</td>
    <td class="tg-0pky">0.2510</td>
  </tr>
    <tr>
    <td class="tg-0pky">full_F</td>
    <td class="tg-0pky">The score obtained using the PACScorer module, which is the arithmetic mean of the above two scores	</td>
    <td class="tg-0pky">0.3908</td>
    <td class="tg-0pky">0.0006854</td>
    <td class="tg-0pky">0.4761</td>
    <td class="tg-0pky">0.2681</td>
  </tr>