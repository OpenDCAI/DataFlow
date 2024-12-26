# 视频数据处理
视频数据的处理主要是基于评估得分的数据集过滤方法。
## 纯视频处理
### 方法分类
|类别描述 | 指标列表| 
|--- |--- |
| 基于视频统计信息 | Motion Score| 
| 基于预训练模型 | FastVQAScorer, FasterVQAScorer, DOVERScorer|

### 方法介绍
| 名称 | 过滤指标 | 过滤维度| 打分器简介 |打分取值范围|  
| ---- | ---- | ---- | ---- | ---- | 
| VideoMotionFilter | Motion Score| 统计|计算帧之间的光流向量的幅度作为评分 |  | 
| [FastVQAFilter](https://arxiv.org/abs/2207.02595v1) | 预训练模型打分 | 模型 | 基于Video Swin Transformer的打分器，加入了Fragment Sampling模块，获得了准确性和速度的提升 | [0,1]| 
| [FasterVQAFilter](https://arxiv.org/abs/2210.05357) | 预训练模型打分 | 模型 | 基于Video Swin Transformer的打分器，在FastVQAScorer的基础上对Fragment Sampling模块进行优化，得到了显著的速度提升 | [0,1] | 
| [DOVERFilter](https://arxiv.org/abs/2211.04894) | 预训练模型打分 | 模型|基于FastVQAScorer的打分器，同时给出了从技术和美学两个角度的评分 || 

## 视频-文本处理

|类别描述 | 指标列表| 
|--- |--- |
| 基于预训练图文模型 | EMScore, PAC-S| 


| 名称 | 过滤指标 | 过滤维度| 打分器简介 |打分取值范围|  
| ---- | ---- | ---- | ---- | ---- |
| [EMScorer](https://arxiv.org/abs/2111.08919) | 基于视频-文本相似度的打分| 模型|基于CLIP的视频-文本打分器，同时支持with-reference和no-reference的打分功能|[0,1] |
| [PACScorer](https://arxiv.org/abs/2303.12112) | 基于视频-文本相似度的打分 | 模型 | 基于CLIP的视频-文本打分器，在EMScore的基础上对CLIP Encoder进行了调优| [0,1] |
