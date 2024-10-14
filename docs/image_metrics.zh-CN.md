# 图像数据评估指标
<!-- 使用`dataflow.list_image_eval_metrics()`打印所有可用的图像评估指标。
```python
import dataflow
dataflow.list_image_eval_metrics()
``` -->

## 纯图像评估指标
### 指标分类
|类别描述|指标列表|
|---|---|
|基于图像统计信息|BRISQUE、ILNIQE、NIQE、PIQE、FID、KID、IS|
|基于神经网络|ARNIQA、TOPIQ、TReS、MANIQA、MUSIQ、DBCNN、PaQ-2-PiQ、HyperIQA、NIMA、WaDIQaM、CNNIQA|
|基于预训练图像-文本模型|Q-Align、CLIPIQA(+)、 LIQE|
### 针对真实图像的评估指标
#### 指标介绍
本仓库调用[pyiqa](https://github.com/chaofengc/IQA-PyTorch)包中的non-reference（NR）算法进行纯图像数据质量评估，各评估指标的介绍可参考[Py-IQA Model Card](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md)。


说明：当同一指标使用了不同训练数据集时，我们使用`指标名-数据集名`进行区分。比如，`arniqa-csiq`中的`csiq`即为数据集名称。当没有标注数据集名时，默认为`koniq`，比如，`arniqa`对应的数据集为`koniq`。
|指标|名称（用于`datagym.get_scorer()`）|评估维度|简介|取值范围|官方仓库或论文|
|---|---|---|---|---|---|
| Q-Align| `qalign` (with quality[default], aesthetic options)|基于预训练图像-文本模型| 使用视觉LLM进行打分。得分越高代表图像质量越高。 |[1,5]|[code](https://github.com/Q-Future/Q-Align)|
| LIQE | `liqe`, `liqe_mix` |基于预训练图像-文本模型| 基于CLIP。得分越高代表图像质量越高。|[1,5]|[code](https://github.com/zwx8981/LIQE)|
| ARNIQA| `arniqa`, `arniqa-live`, `arniqa-csiq`, `arniqa-tid`, `arniqa-kadid`, `arniqa-clive`, `arniqa-flive`, `arniqa-spaq` |基于神经网络|学习图像失真流形。得分越高代表图像质量越高。||[paper](https://arxiv.org/abs/2310.14918)|
| TOPIQ | `topiq_nr`, `topiq_nr-flive`, `topiq_nr-spaq` |基于神经网络| 基于语义的自顶向下图像质量评估。得分越高代表图像质量越高。|[0,1]|[paper](https://arxiv.org/abs/2308.03060)|
| TReS | `tres`, `tres-flive` |基于神经网络|通过相对排名和自我一致性增强指标的鲁棒性。得分越高代表图像质量越高。|[0,100]|[code](https://github.com/isalirezag/TReS)|
| CLIPIQA(+) |`clipiqa`, `clipiqa+`, `clipiqa+_vitL14_512`,`clipiqa+_rn50_512`|基于预训练图像-文本模型| 基于CLIP设计Antonym prompt pairing（反义提示词对）。使用了不同骨干网络的CLIPIQA(+)，默认为RN50。得分越高代表图像质量越高。|[0,1]|[code](https://github.com/IceClear/CLIP-IQA)|
| MANIQA | `maniqa`, `maniqa-kadid`, `maniqa-pipal` |基于神经网络|设计了多维注意力网络用于质量评估。得分越高代表图像质量越高。||[paper](https://arxiv.org/abs/2204.08958)|
| MUSIQ| `musiq`, `musiq-spaq`, `musiq-paq2piq`, `musiq-ava` |基于神经网络| 设计了多尺度图像质量评估Transformer。得分越高代表图像质量越高。||[paper](https://arxiv.org/abs/2108.05997)|
| DBCNN| `dbcnn` |基于神经网络|设计了双线性模型来处理合成失真和真实失真。得分越高代表图像质量越高。||[paper](https://ieeexplore.ieee.org/document/8576582)|
| PaQ-2-PiQ| `paq2piq` |基于神经网络|设计了产生全局到局部推断以及局部到全局推断的质量评估结构。得分越高代表图像质量越高。||[code](https://baidut.github.io/PaQ-2-PiQ/)|
| HyperIQA |`hyperiqa` |基于神经网络|设计了自适应超网络架构以处理真实世界的图像失真。得分越高代表图像质量越高。||[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf)|
| NIMA |`nima`, `nima-vgg16-ava` |基于神经网络| 使用卷积神经网络预测人类意见得分的**分布**。。得分越高代表图像质量越高。||[paper](https://arxiv.org/abs/1709.05424)|
| WaDIQaM| `wadiqam_nr` |基于神经网络|基于卷积神经网络。得分越高代表图像质量越高。||[paper](https://ieeexplore.ieee.org/abstract/document/8063957)|
| CNNIQA |`cnniqa` |基于神经网络|基于卷积神经网络。得分越高代表图像质量越高。||[paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)|
| NRQM(Ma)<sup>[2](#fn2)</sup> |`nrqm` |超分辨率图像评估|基于图像统计信息||[paper](https://arxiv.org/abs/1612.05890)|
| PI(Perceptual Index) |`pi` |超分辨率图像评估|基于Ma's score和NIQE。得分越低代表图像质量越高。||[paper](https://arxiv.org/abs/1711.06077)|
| BRISQUE| `brisque`, `brisque_matlab` |基于图像统计信息|在空间域中进进行评估；计算复杂度低。得分越低代表图像质量越高。||[paper](https://ieeexplore.ieee.org/document/6272356)|
| ILNIQE | `ilniqe` |基于图像统计信息|基于自然图像统计特征。得分越低代表图像质量越高。||[paper](https://ieeexplore.ieee.org/document/7094273)|
| NIQE | `niqe`, `niqe_matlab` |基于图像统计信息|基于自然、未失真的图像数据的统计特征。得分越低代表图像质量越高。||[paper](https://ieeexplore.ieee.org/document/6353522)|
| PIQE | `piqe` |基于图像统计信息|提取局部特征来预测质量；计算复杂度低。得分越低代表图像质量越高。||[paper](https://ieeexplore.ieee.org/document/7084843)|



#### 参考值
为更好的提供数据质量参考，我们使用以上指标对MSCOCO 2017 train进行评估，得到的指标数值分布如下:
<!--<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style> -->
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">指标</th>
    <th class="tg-0pky">名称</th>
    <th class="tg-0pky">均值</th>
    <th class="tg-0pky">方差</th>
    <th class="tg-0pky">最大值</th>
    <th class="tg-0pky">最小值</th>
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

### 针对生成图像的评估指标
| 指标 | 名称（用于`datagym.get_scorer()`） | 评估维度 | 简介 | 取值范围 | 官方仓库或论文 |
|------|-----------------------------------|-----------|------|------------|----------------|
| FID | `fid_score` | 生成图像与真实图像间的统计差异 | 使用Inception网络计算特征，进而计算两个数据集的统计距离，评估生成模型的质量。 | 最佳值为0，较低的值表明较小的差异和更高的图像质量，无上限 |[paper](https://arxiv.org/pdf/1706.08500)|
| KID | `kid_score` | 生成图像的无偏质量估计 | Kernel Inception Distance，使用Inception网络特征计算MMD，提供对生成图像质量的无偏估计。 | 最佳值为0，较低的值表示更低的偏差和更好的图像质量，无上限 |[paper](https://arxiv.org/abs/1801.01401) |
| IS | `is_score` | 生成图像的多样性和清晰度 | 通过计算生成图像的Inception网络输出的信息熵，评估图像多样性及清晰度。 | 值越高表示图像质量越好，通常分数在1到10之间，但无具体上限 | [paper](https://arxiv.org/pdf/1606.03498) |
#### 参考值
为更好的提供数据质量参考，我们使用了四种模型：flux-dev, flux-schnell, stable-diffusion-3-medium 和 sdxl，对在 LLaVA Pretrain 数据集上随机选取的500个 image-caption 对进行测试。每个模型根据给定的 caption 生成相应的图片，并通过上述三个指标对生成图片的质量进行全面评估。结果如下：
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">模型名称</th>
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
stable-diffusion-3-medium在FID指标上表现最佳，表明其生成的图像与真实图像在统计特征上最为接近。而在IS评分中，flux-dev表现最优，显示了较高的图像多样性和清晰度。KID结果中，stable-diffusion-3-medium同样表现较好，表示其图像质量的偏差较小。

## 图像-文本评估指标
### 图文对齐指标
指标数值越高，则image-caption对齐程度越好。
|指标|名称（用于`datagym.get_scorer()`|数据类型|简介|取值范围|官方仓库或论文|
|---|---|---|---|---|---|
|CLIP|`clip`|image-caption|经典的图文对齐分数。数值越大，对齐程度越高。|[0,1]|[code](https://github.com/openai/CLIP)|
|LongCLIP|`longclip`|image-caption|可输入更长文本、粒度更细的CLIP。数值越大，对齐程度越高。|[0,1]|[code](https://github.com/beichenzbc/Long-CLIP)|
|FLEUR|`fleur`|image-caption|使用LLaVA模型进行评分。数值越大，对齐程度越高。|[0,1]|[code](https://github.com/Yebin46/FLEUR)|
|VQA Score|`vqa_score`|image-caption|使用CLIP-FlanT5模型进行打分。数值越大，对齐程度越高。|[0,1]|[code](https://github.com/linzhiqiu/t2v_metrics)|

### 图文SFT数据评估指标

|指标名称|评估维度|数据类型|简介|取值范围|官方仓库或论文|
|---|---|---|---|---|---|
|vqa_score|图片-对话对齐度|image-dialog|使用LLaVA模型判断对话正误。数值越大，对齐程度越高。|(-∞,0] |/|
