# 视频数据评估指标

## 纯视频评估指标
### 指标分类
|类别描述 | 指标列表| 
|--- |--- |
| 基于视频统计信息 | Motion Score| 
| 基于预训练模型 | FastVQAScorer, FasterVQAScorer, DOVERScorer|

### 指标介绍
| 名称 | 评估指标 | 评估维度| 简介 |取值范围|  
| ---- | ---- | ---- | ---- | ---- | 
| VideoMotionScorer | Motion Score| 统计|计算帧之间的光流向量的幅度作为评分 |  | 
| [FastVQAScorer](https://arxiv.org/abs/2207.02595v1) | 预训练模型打分 | 模型 | 基于Video Swin Transformer的打分器，加入了Fragment Sampling模块，获得了准确性和速度的提升 | [0,1]| 
| [FasterVQAScorer](https://arxiv.org/abs/2210.05357) | 预训练模型打分 | 模型 | 基于Video Swin Transformer的打分器，在FastVQAScorer的基础上对Fragment Sampling模块进行优化，得到了显著的速度提升 | [0,1] | 
| [DOVERScorer](https://arxiv.org/abs/2211.04894) | 预训练模型打分 | 模型|基于FastVQAScorer的打分器，同时给出了从技术和美学两个角度的评分 || 

### 参考值
为更好的提供数据质量参考，我们使用以上指标对KoNViD-1k数据集进行评估，得到的指标数值分布如下:
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">指标</th>
    <th class="tg-0pky">名称</th>
    <th class="tg-0pky">简介</th>
    <th class="tg-0pky">均值</th>
    <th class="tg-0pky">方差</th>
    <th class="tg-0pky">最大值</th>
    <th class="tg-0pky">最小值</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">Motion Score</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">计算视频帧之间的光流向量的幅度作为评分，视频中的运动越强烈，帧之间的变化越大，分数越高</td>
    <td class="tg-0pky">6.2745</td>
    <td class="tg-0pky">19.28</td>
    <td class="tg-0pky">25.23</td>
    <td class="tg-0pky">0.001623</td>
  </tr>
  <tr>
    <td class="tg-0pky" >FastVQA</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">使用FastVQAScorer模块得到的评分，视频质量越好，分数越高</td>
    <td class="tg-0pky">0.4987</td>
    <td class="tg-0pky">0.04554</td>
    <td class="tg-0pky">0.9258</td>
    <td class="tg-0pky">0.007619</td>
  </tr>
  <tr>
    <td class="tg-0pky">FasterVQA</td>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">使用FasterVQAScorer模块得到的评分，视频质量越好，分数越高</td>
    <td class="tg-0pky">0.5134</td>
    <td class="tg-0pky">0.04558</td>
    <td class="tg-0pky">0.9066</td>
    <td class="tg-0pky">0.03686</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">DOVER</td>
    <td class="tg-0pky">technical</td>
    <td class="tg-0pky">使用DOVERScorer模块得到的评分之一，视频在技术方面的质量越好，分数越高</td>
    <td class="tg-0pky">-0.1107</td>
    <td class="tg-0pky">0.001755</td>
    <td class="tg-0pky">-0.006550</td>
    <td class="tg-0pky">-0.3175</td>
  </tr>
  <tr>
    <td class="tg-0pky">aesthetic</td>
    <td class="tg-0pky">使用DOVERScorer模块得到的评分之一，视频在美学方面的质量越好，分数越高</td>
    <td class="tg-0pky">-0.008419</td>
    <td class="tg-0pky">0.004569</td>
    <td class="tg-0pky">0.1869</td>
    <td class="tg-0pky">-0.2629</td>
  </tr>
</tbody></table>

<!-- - VideoMotionScorer: 计算视频的Motion Score作为评分
- FastVQAScorer: ECCV 2022 论文 [FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling](https://arxiv.org/abs/2207.02595v1)所提出的基于Video Swin Transformer的打分器。
- FasterVQAScorer: TPAMI 2023 论文 [Neighbourhood Representative Sampling for Efficient End-to-end Video Quality Assessment](https://arxiv.org/abs/2210.05357) 所提出的在FastVQAScorer扩展的打分器。
- DOVERScorer: ICCV 2023 论文 [Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives](https://arxiv.org/abs/2211.04894) 提出的基于FastVQAScorer的打分器，同时给出了从技术和美学两个角度的评分 -->

## 视频-文本评估指标
|类别描述 | 指标列表| 
|--- |--- |
| 基于预训练图文模型 | EMScore, PAC-S| 


| 名称 | 评估指标 |评估维度|简介 | 取值范围|
| ---- | ---- | ---- | ---- | ---- |
| [EMScorer](https://arxiv.org/abs/2111.08919) | 基于视频-文本相似度的打分| 模型|基于CLIP的视频-文本打分器，同时支持with-reference和no-reference的打分功能|[0,1] |
| [PACScorer](https://arxiv.org/abs/2303.12112) | 基于视频-文本相似度的打分 | 模型 | 基于CLIP的视频-文本打分器，在EMScore的基础上对CLIP Encoder进行了调优| [0,1] |

### 参考值
为更好的提供数据质量参考，我们使用以上指标对VATEX数据集([链接](https://huggingface.co/datasets/lmms-lab/VATEX))进行评估，得到的指标数值分布如下:
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">指标</th>
    <th class="tg-0pky">名称</th>
    <th class="tg-0pky">简介</th>
    <th class="tg-0pky">均值</th>
    <th class="tg-0pky">方差</th>
    <th class="tg-0pky">最大值</th>
    <th class="tg-0pky">最小值</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">EMScorer</td>
    <td class="tg-0pky">figr_F</td>
    <td class="tg-0pky">使用EMScorer模块得到的评分，视频和文本在细粒度层面的相似度越高，分数越高</td>
    <td class="tg-0pky">0.2712</td>
    <td class="tg-0pky">0.0003667</td>
    <td class="tg-0pky">0.3461</td>
    <td class="tg-0pky">0.1987</td>
  </tr>
  <tr>
    <td class="tg-0pky">cogr</td>
    <td class="tg-0pky">使用EMScorer模块得到的评分，视频和文本在粗粒度层面的相似度越高，分数越高</td>
    <td class="tg-0pky">0.3106</td>
    <td class="tg-0pky">0.0009184</td>
    <td class="tg-0pky">0.4144</td>
    <td class="tg-0pky">0.18</td>
  </tr>
    <tr>
    <td class="tg-0pky">full_F</td>
    <td class="tg-0pky">使用EMScorer模块得到的评分，取以上两种评分的算术平均数</td>
    <td class="tg-0pky">0.2909</td>
    <td class="tg-0pky">0.0005776</td>
    <td class="tg-0pky">0.3712</td>
    <td class="tg-0pky">0.3807</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">PACScorer</td>
    <td class="tg-0pky">figr_F</td>
    <td class="tg-0pky">使用PACScorer模块得到的评分，视频和文本在细粒度层面的相似度越高，分数越高</td>
    <td class="tg-0pky">0.36553</td>
    <td class="tg-0pky">0.0004902</td>
    <td class="tg-0pky">0.4456</td>
    <td class="tg-0pky">0.2778</td>
  </tr>
  <tr>
    <td class="tg-0pky">cogr</td>
    <td class="tg-0pky">使用PACScorer模块得到的评分，视频和文本在粗粒度层面的相似度越高，分数越高</td>
    <td class="tg-0pky">0.4160</td>
    <td class="tg-0pky">0.001021</td>
    <td class="tg-0pky">0.5222</td>
    <td class="tg-0pky">0.2510</td>
  </tr>
    <tr>
    <td class="tg-0pky">full_F</td>
    <td class="tg-0pky">使用PACScorer模块得到的评分，取以上两种评分的算术平均数</td>
    <td class="tg-0pky">0.3908</td>
    <td class="tg-0pky">0.0006854</td>
    <td class="tg-0pky">0.4761</td>
    <td class="tg-0pky">0.2681</td>
  </tr>