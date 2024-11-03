# Image Data Evaluation Metrics

<!-- Use `dataflow.list_image_eval_metrics()` to print all available image evaluation metrics.
```python
import dataflow
dataflow.list_image_eval_metrics()
``` -->

## Pure Image Evaluation Metrics
### Metric Classification
| Category Description | Metric List |
|---|---|
| Based on Image Statistics | BRISQUE, ILNIQE, NIQE, PIQE, FID, KID, IS |
| Based on Neural Networks | ARNIQA, TOPIQ, TReS, MANIQA, MUSIQ, DBCNN, PaQ-2-PiQ, HyperIQA, NIMA, WaDIQaM, CNNIQA |
| Based on Pre-trained Image-Text Models | Q-Align, CLIPIQA(+), LIQE |

### Evaluation Metrics for Real Images
#### Metric Introduction
This repository uses non-reference (NR) algorithms from the [pyiqa](https://github.com/chaofengc/IQA-PyTorch) package for pure image data quality assessment. Introductions to each evaluation metric can be found in the [Py-IQA Model Card](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md).

Note: When the same metric uses different training datasets, we distinguish them using `Metric Name-Dataset Name`. For example, `arniqa-csiq` uses `csiq` as the dataset name. When the dataset name is not specified, it defaults to `koniq`, such as `arniqa` which corresponds to the `koniq` dataset.

| Metric | Name (for `datagym.get_scorer()`) | Evaluation Dimension | Introduction | Value Range | Official Repository or Paper |
|---|---|---|---|---|---|
| Q-Align | `qalign` (with quality[default], aesthetic options) | Based on Pre-trained Image-Text Model | Scoring using Visual LLM. The larger the value, the higher the quality.| | [1,5] | [code](https://github.com/Q-Future/Q-Align) | 
| LIQE | `liqe`, `liqe_mix` | Based on Pre-trained Image-Text Model | Based on CLIP. The larger the value, the higher the quality. | [1,5] | [code](https://github.com/zwx8981/LIQE) | 
| ARNIQA | `arniqa`, `arniqa-live`, `arniqa-csiq`, `arniqa-tid`, `arniqa-kadid`, `arniqa-clive`, `arniqa-flive`, `arniqa-spaq` | Based on Neural Networks | Learning the Image Distortion Manifold. The larger the value, the higher the quality. | | [paper](https://arxiv.org/abs/2310.14918) | 
| TOPIQ | `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq` | Based on Neural Networks | Semantic-based Top-down Image Quality Assessment. The larger the value, the higher the quality. | [0,1] | [paper](https://arxiv.org/abs/2308.03060) | 
| TReS | `tres`, `tres-flive` | Based on Neural Networks | Enhancing Metric Robustness through Relative Ranking and Self-consistency. The larger the value, the higher the quality. | [0,100] | [code](https://github.com/isalirezag/TReS) | 
| CLIPIQA(+) | `clipiqa`, `clipiqa+`, `clipiqa+_vitL14_512`, `clipiqa+_rn50_512` | Based on Pre-trained Image-Text Model | Based on CLIP with Antonym prompt pairing. CLIPIQA(+) with different backbone networks, default is RN50. The larger the value, the higher the quality. | [0,1] | [code](https://github.com/IceClear/CLIP-IQA) | 
| MANIQA | `maniqa`, `maniqa-kadid`, `maniqa-pipal` | Based on Neural Networks | Designed a Multi-dimension Attention Network for Quality Assessment. The larger the value, the higher the quality. | | [paper](https://arxiv.org/abs/2204.08958) | 
| MUSIQ | `musiq`, `musiq-spaq`, `musiq-paq2piq`, `musiq-ava` | Based on Neural Networks | Designed a Multi-scale Image Quality Assessment Transformer. The larger the value, the higher the quality. | | [paper](https://arxiv.org/abs/2108.05997) | 
| DBCNN | `dbcnn` | Based on Neural Networks | Designed a Bilinear Model to Address Synthetic and Realistic Distortions. The larger the value, the higher the quality. | | [paper](https://ieeexplore.ieee.org/document/8576582) | 
| PaQ-2-PiQ | `paq2piq` | Based on Neural Networks | Designed a Quality Assessment Structure that Generates Global-to-Local and Local-to-Global Inferences. The larger the value, the higher the quality. | | [code](https://baidut.github.io/PaQ-2-PiQ/) | 
| HyperIQA | `hyperiqa` | Based on Neural Networks | Designed an Adaptive Hypernetwork Architecture to Handle Realistic Image Distortions. The larger the value, the higher the quality. | | [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) | 
| NIMA | `nima`, `nima-vgg16-ava` | Based on Neural Networks | Predicting Human Opinion Scores using Convolutional Neural Networks **Distribution**. The larger the value, the higher the quality. | | [paper](https://arxiv.org/abs/1709.05424) | 
| WaDIQaM | `wadiqam_nr` | Based on Neural Networks | Based on Convolutional Neural Networks. The larger the value, the higher the quality. | | [paper](https://ieeexplore.ieee.org/abstract/document/8063957) | 
| CNNIQA | `cnniqa` | Based on Neural Networks | Based on Convolutional Neural Networks. The larger the value, the higher the quality. | | [paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) | 
| NRQM(Ma)^2^ | `nrqm` | Super-Resolution Image Assessment | Based on Image Statistics. The smaller the value, the higher the quality. | | [paper](https://arxiv.org/abs/1612.05890) | 
| PI (Perceptual Index) | `pi` | Super-Resolution Image Assessment | Based on Ma's score and NIQE. The smaller the value, the higher the quality.  | | [paper](https://arxiv.org/abs/1711.06077) | 
| BRISQUE | `brisque`, `brisque_matlab` | Based on Image Statistics | Assessed in the Spatial Domain; Low Computational Complexity. The smaller the value, the higher the quality.  | | [paper](https://ieeexplore.ieee.org/document/6272356) | 
| ILNIQE | `ilniqe` | Based on Image Statistics | Based on Natural Image Statistical Features. The smaller the value, the higher the quality.  | | [paper](https://ieeexplore.ieee.org/document/7094273) | 
| NIQE | `niqe`, `niqe_matlab` | Based on Image Statistics | Based on Statistical Features of Natural, Undistorted Image Data. The smaller the value, the higher the quality.  | | [paper](https://ieeexplore.ieee.org/document/6353522) | 
| PIQE | `piqe` | Based on Image Statistics | Extract Local Features to Predict Quality; Low Computational Complexity. The smaller the value, the higher the quality.  | | [paper](https://ieeexplore.ieee.org/document/7084843) | 

#### Reference Values
To better provide data quality references, we have evaluated the MSCOCO 2017 train using the above metrics, and the distribution of metric values obtained is as follows:

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Metric</th>
    <th class="tg-0pky">Name</th>
    <th class="tg-0pky">Mean</th>
    <th class="tg-0pky">Variance</th>
    <th class="tg-0pky">Maximum</th>
    <th class="tg-0pky">Minimum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Q-Align</td>
    <td class="tg-0pky">qalign</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">LIQE</td>
    <td class="tg-0pky">liqe</td>
    <td class="tg-0pky">4.152</td>
    <td class="tg-0pky">1.004</td>
    <td class="tg-0pky">5.000</td>
    <td class="tg-0pky">1.000</td>
  </tr>
  <tr>
    <td class="tg-0pky">liqe_mix</td>
    <td class="tg-0pky">4.090</td>
    <td class="tg-0pky">0.893</td>
    <td class="tg-0pky">5.000</td>
    <td class="tg-0pky">1.000</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="9">ARNIQA</td>
    <td class="tg-0pky">arniqa</td>
    <td class="tg-0pky">0.705</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.867</td>
    <td class="tg-0pky">0.150</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-clive</td>
    <td class="tg-0pky">0.649</td>
    <td class="tg-0pky">0.103</td>
    <td class="tg-0pky">0.961</td>
    <td class="tg-0pky">-0.105</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-csiq</td>
    <td class="tg-0pky">0.900</td>
    <td class="tg-0pky">0.073</td>
    <td class="tg-0pky">1.081</td>
    <td class="tg-0pky">0.319</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-flive</td>
    <td class="tg-0pky">0.724</td>
    <td class="tg-0pky">0.036</td>
    <td class="tg-0pky">0.838</td>
    <td class="tg-0pky">0.097</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-kadid</td>
    <td class="tg-0pky">0.635</td>
    <td class="tg-0pky">0.122</td>
    <td class="tg-0pky">0.965</td>
    <td class="tg-0pky">-0.013</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-koniq</td>
    <td class="tg-0pky">0.705</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.867</td>
    <td class="tg-0pky">0.150</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-live</td>
    <td class="tg-0pky">0.788</td>
    <td class="tg-0pky">0.069</td>
    <td class="tg-0pky">0.958</td>
    <td class="tg-0pky">0.227</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-spqa</td>
    <td class="tg-0pky">0.699</td>
    <td class="tg-0pky">0.104</td>
    <td class="tg-0pky">1.100</td>
    <td class="tg-0pky">0.056</td>
  </tr>
  <tr>
    <td class="tg-0pky">arniqa-tid</td>
    <td class="tg-0pky">0.548</td>
    <td class="tg-0pky">0.081</td>
    <td class="tg-0pky">0.803</td>
    <td class="tg-0pky">0.140</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">TOPIQ</td>
    <td class="tg-0pky">topiq_nr</td>
    <td class="tg-0pky">0.610</td>
    <td class="tg-0pky">0.116</td>
    <td class="tg-0pky">0.851</td>
    <td class="tg-0pky">0.073</td>
  </tr>
  <tr>
    <td class="tg-0pky">topiq_iaa_res50</td>
    <td class="tg-0pky">5.013</td>
    <td class="tg-0pky">0.492</td>
    <td class="tg-0pky">6.969</td>
    <td class="tg-0pky">2.812</td>
  </tr>
  <tr>
    <td class="tg-0pky">topiq_iaa</td>
    <td class="tg-0pky">4.838</td>
    <td class="tg-0pky">0.539</td>
    <td class="tg-0pky">7.129</td>
    <td class="tg-0pky">2.607</td>
  </tr>
  <tr>
    <td class="tg-0pky">topiq_nr-flive</td>
    <td class="tg-0pky">0.728</td>
    <td class="tg-0pky">0.036</td>
    <td class="tg-0pky">0.825</td>
    <td class="tg-0pky">0.371</td>
  </tr>
  <tr>
    <td class="tg-0pky">topiq_nr-spaq</td>
    <td class="tg-0pky">0.679</td>
    <td class="tg-0pky">0.102</td>
    <td class="tg-0pky">0.930</td>
    <td class="tg-0pky">0.119</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">CLIPIQA(+)</td>
    <td class="tg-0pky">clipiqa</td>
    <td class="tg-0pky">0.622</td>
    <td class="tg-0pky">0.149</td>
    <td class="tg-0pky">0.934</td>
    <td class="tg-0pky">0.056</td>
  </tr>
  <tr>
    <td class="tg-0pky">clipiqa+</td>
    <td class="tg-0pky">0.659</td>
    <td class="tg-0pky">0.100</td>
    <td class="tg-0pky">0.918</td>
    <td class="tg-0pky">0.130</td>
  </tr>
  <tr>
    <td class="tg-0pky">clipiqa+_rn50_512</td>
    <td class="tg-0pky">0.571</td>
    <td class="tg-0pky">0.122</td>
    <td class="tg-0pky">0.883</td>
    <td class="tg-0pky">0.050</td>
  </tr>
  <tr>
    <td class="tg-0pky">clipiqa+_vitL14_512</td>
    <td class="tg-0pky">0.593</td>
    <td class="tg-0pky">0.128</td>
    <td class="tg-0pky">0.893</td>
    <td class="tg-0pky">0.077</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">MANIQA</td>
    <td class="tg-0pky">maniqa</td>
    <td class="tg-0pky">0.454</td>
    <td class="tg-0pky">0.106</td>
    <td class="tg-0pky">0.789</td>
    <td class="tg-0pky">0.021</td>
  </tr>
  <tr>
    <td class="tg-0pky">maniqa-kadid</td>
    <td class="tg-0pky">0.637</td>
    <td class="tg-0pky">0.122</td>
    <td class="tg-0pky">0.877</td>
    <td class="tg-0pky">0.075</td>
  </tr>
  <tr>
    <td class="tg-0pky">maniqa-koniq</td>
    <td class="tg-0pky">0.454</td>
    <td class="tg-0pky">0.106</td>
    <td class="tg-0pky">0.789</td>
    <td class="tg-0pky">0.021</td>
  </tr>
  <tr>
    <td class="tg-0pky">maniqa-pipal</td>
    <td class="tg-0pky">0.676</td>
    <td class="tg-0pky">0.062</td>
    <td class="tg-0pky">0.888</td>
    <td class="tg-0pky">0.228</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">MUSIQ</td>
    <td class="tg-0pky">musiq</td>
    <td class="tg-0pky">69.086</td>
    <td class="tg-0pky">7.833</td>
    <td class="tg-0pky">79.559</td>
    <td class="tg-0pky">12.679</td>
  </tr>
  <tr>
    <td class="tg-0pky">musiq-ava</td>
    <td class="tg-0pky">4.939</td>
    <td class="tg-0pky">0.546</td>
    <td class="tg-0pky">7.269</td>
    <td class="tg-0pky">2.434</td>
  </tr>
  <tr>
    <td class="tg-0pky">musiq-koniq</td>
    <td class="tg-0pky">69.086</td>
    <td class="tg-0pky">7.833</td>
    <td class="tg-0pky">79.559</td>
    <td class="tg-0pky">12.679</td>
  </tr>
  <tr>
    <td class="tg-0pky">musiq-paq2piq</td>
    <td class="tg-0pky">72.792</td>
    <td class="tg-0pky">3.520</td>
    <td class="tg-0pky">79.772</td>
    <td class="tg-0pky">39.551</td>
  </tr>
  <tr>
    <td class="tg-0pky">musiq-spaq</td>
    <td class="tg-0pky">70.534</td>
    <td class="tg-0pky">8.661</td>
    <td class="tg-0pky">81.385</td>
    <td class="tg-0pky">14.290</td>
  </tr>
  <tr>
    <td class="tg-0pky">DBCNN</td>
    <td class="tg-0pky">dbcnn</td>
    <td class="tg-0pky">0.634</td>
    <td class="tg-0pky">0.100</td>
    <td class="tg-0pky">0.834</td>
    <td class="tg-0pky">0.143</td>
  </tr>
  <tr>
    <td class="tg-0pky">PaQ-2-PiQ</td>
    <td class="tg-0pky">paq2piq</td>
    <td class="tg-0pky">74.669</td>
    <td class="tg-0pky">3.731</td>
    <td class="tg-0pky">85.906</td>
    <td class="tg-0pky">15.859</td>
  </tr>
  <tr>
    <td class="tg-0pky">HyperIQA</td>
    <td class="tg-0pky">hyperiqa</td>
    <td class="tg-0pky">0.618</td>
    <td class="tg-0pky">0.105</td>
    <td class="tg-0pky">0.843</td>
    <td class="tg-0pky">0.082</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">NIMA</td>
    <td class="tg-0pky">nima</td>
    <td class="tg-0pky">4.941</td>
    <td class="tg-0pky">0.537</td>
    <td class="tg-0pky">7.056</td>
    <td class="tg-0pky">2.463</td>
  </tr>
  <tr>
    <td class="tg-0pky">nima-koniq</td>
    <td class="tg-0pky">0.654</td>
    <td class="tg-0pky">0.084</td>
    <td class="tg-0pky">0.849</td>
    <td class="tg-0pky">0.048</td>
  </tr>
  <tr>
    <td class="tg-0pky">nima-spaq</td>
    <td class="tg-0pky">71.036</td>
    <td class="tg-0pky">10.099</td>
    <td class="tg-0pky">98.191</td>
    <td class="tg-0pky">12.237</td>
  </tr>
  <tr>
    <td class="tg-0pky">nima-vgg-ava</td>
    <td class="tg-0pky">5.040</td>
    <td class="tg-0pky">0.503</td>
    <td class="tg-0pky">7.327</td>
    <td class="tg-0pky">2.374</td>
  </tr>
  <tr>
    <td class="tg-0pky">WaDIQaM</td>
    <td class="tg-0pky">wadiqam_nr</td>
    <td class="tg-0pky">-0.066</td>
    <td class="tg-0pky">0.207</td>
    <td class="tg-0pky">0.377</td>
    <td class="tg-0pky">-1.281</td>
  </tr>
  <tr>
    <td class="tg-0pky">CNNIQA</td>
    <td class="tg-0pky">cnniqa</td>
    <td class="tg-0pky">0.655</td>
    <td class="tg-0pky">0.070</td>
    <td class="tg-0pky">0.759</td>
    <td class="tg-0pky">0.089</td>
  </tr>
  <tr>
    <td class="tg-0pky">NRQM(Ma)^2^</td>
    <td class="tg-0pky">nrqm</td>
    <td class="tg-0pky">8.050</td>
    <td class="tg-0pky">1.001</td>
    <td class="tg-0pky">9.222</td>
    <td class="tg-0pky">1.600</td>
  </tr>
  <tr>
    <td class="tg-0pky">PI</td>
    <td class="tg-0pky">pi</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">BRISQUE</td>
    <td class="tg-0pky">brisque</td>
    <td class="tg-0pky">13.777</td>
    <td class="tg-0pky">11.891</td>
    <td class="tg-0pky">184.089</td>
    <td class="tg-0pky">-67.742</td>
  </tr>
  <tr>
    <td class="tg-0pky">ILNIQE</td>
    <td class="tg-0pky">ilniqe</td>
    <td class="tg-0pky">22.919</td>
    <td class="tg-0pky">6.589</td>
    <td class="tg-0pky">154.256</td>
    <td class="tg-0pky">12.733</td>
  </tr>
  <tr>
    <td class="tg-0pky">NIQE</td>
    <td class="tg-0pky">niqe</td>
    <td class="tg-0pky">3.718</td>
    <td class="tg-0pky">1.082</td>
    <td class="tg-0pky">55.155</td>
    <td class="tg-0pky">1.430</td>
  </tr>
  <tr>
    <td class="tg-0pky">PIQE</td>
    <td class="tg-0pky">piqe</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky"></td>
  </tr>
</tbody></table>


### Evaluation Metrics for Generated Images

| Metric | Name for `datagym.get_scorer()` | Evaluation Dimension | Description | Range | Official Repository or Paper |
|--------|---------------------------------|---------------------|-------------|------|-----------------------------|
| FID    | `fid_score`                     | Statistical difference between generated and real images | Uses Inception network to calculate features and then the statistical distance between two datasets to evaluate the quality of generative models. | Best value is 0, lower values indicate smaller differences and higher image quality, no upper limit | [paper](https://arxiv.org/pdf/1706.08500) |
| KID    | `kid_score`                     | Unbiased quality estimation of generated images | Kernel Inception Distance, uses Inception network features to calculate MMD, providing an unbiased estimation of the quality of generated images. | Best value is 0, lower values indicate lower bias and better image quality, no upper limit | [paper](https://arxiv.org/abs/1801.01401) |
| IS     | `is_score`                     | Diversity and clarity of generated images | Evaluates the diversity and clarity of images by calculating the entropy of the Inception network's output. | Higher values indicate better image quality, typically scores range from 1 to 10, but no specific upper limit | [paper](https://arxiv.org/pdf/1606.03498) |
#### Reference Values
To better provide data quality references, we used four models: flux-dev, flux-schnell, stable-diffusion-3-medium, and sdxl, to test 500 randomly selected image-caption pairs from the LLaVA Pretrain dataset. Each model generated images based on the given captions, and their quality was comprehensively assessed using the following three indicators. Here are the results:
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Model Name</th>
    <th class="tg-0pky">Inception Score (IS)</th>
    <th class="tg-0pky">Fréchet Inception Distance (FID)</th>
    <th class="tg-0pky">Kernel Inception Distance (KID)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">flux-dev</td>
    <td class="tg-0pky">7.195 ± 0.809</td>
    <td class="tg-0pky">101.572</td>
    <td class="tg-0pky">0.00903 ± 0.00069</td>
  </tr>
  <tr>
    <td class="tg-0pky">flux-schnell</td>
    <td class="tg-0pky">6.193 ± 0.546</td>
    <td class="tg-0pky">102.739</td>
    <td class="tg-0pky">0.00667 ± 0.00055</td>
  </tr>
  <tr>
    <td class="tg-0pky">stable-diffusion-3-medium</td>
    <td class="tg-0pky">6.740 ± 0.582</td>
    <td class="tg-0pky">100.235</td>
    <td class="tg-0pky">0.00609 ± 0.00056</td>
  </tr>
  <tr>
    <td class="tg-0pky">sdxl</td>
    <td class="tg-0pky">6.809 ± 0.994</td>
    <td class="tg-0pky">112.807</td>
    <td class="tg-0pky">0.01051 ± 0.00065</td>
  </tr>
</tbody></table>
Stable-diffusion-3-medium performed the best in terms of FID, indicating that its generated images are statistically closest to real images. Flux-dev showed the best results in the IS score, reflecting higher diversity and clarity in images. Similarly, stable-diffusion-3-medium also exhibited superior performance in KID results, indicating a smaller deviation in image quality.

### Image-Text Evaluation Metrics

#### Image-Text Alignment Metrics
Higher metric values indicate better alignment between images and captions.

| Metric  | Name for `datagym.get_scorer()` | Data Type     | Description | Range  |Official Repository or Paper |
|---------|--------------------------------|--------------|-------------|--------|-----------------------------|
| CLIP    | `clip`                         | image-caption | Classic image-text alignment score. The larger the value, the higher the alignment degree. | [0,1]  | [code](https://github.com/openai/CLIP) |
| LongCLIP| `longclip`                     | image-caption | CLIP with longer text input and finer granularity. The larger the value, the higher the alignment degree. | [0,1]  | [code](https://github.com/beichenzbc/Long-CLIP) |
| FLEUR   | `fleur`                       | image-caption | Scores using the LLaVA model. The larger the value, the higher the alignment degree. | [0,1]  | [code](https://github.com/Yebin46/FLEUR) |
| VQA Score| `vqa_score`                   | image-caption | Scores using the CLIP-FlanT5 model. The larger the value, the higher the alignment degree. | [0,1]  | [code](https://github.com/linzhiqiu/t2v_metrics) |

#### SFT Data Evaluation Metrics for Image-Text

| Metric Name | Evaluation Dimension | Data Type       | Description | Range | Official Repository or Paper |
|-------------|---------------------|----------------|-------------|------|-----------------------------|
| visual_dialog_score   | Image-dialog alignment |image-dialog | Use the LLaVA model to judge the correctness of the dialogue. The larger the value, the higher the alignment degree.| (-∞,0] |/|
