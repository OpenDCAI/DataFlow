# 图像数据处理器介绍
注：我们为每个打分器都设置了对应的过滤器，命名规则为```MetricnameScorer```与```MetriFilter```，这部分过滤器的介绍请参考[image_metrics.zh-CN.md](./image_metrics.zh-CN.md)。以下仅介绍与之前的打分器不同的处理器。


|处理器名称|简介|
|---|---|
|ImagePHashDeduplicator|基于[imagededup](https://github.com/idealo/imagededup)的[Perceptual hashing](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)方法|
|ImageDHashDeduplicator|基于[imagededup](https://github.com/idealo/imagededup)的[Difference hashing](https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)方法|
|ImageWHashDeduplicator|基于[imagededup](https://github.com/idealo/imagededup)的[Wavelet hashing](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)方法|
|ImageAHashDeduplicator|基于[imagededup](https://github.com/idealo/imagededup)的[Average hashing](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)|
|ImageAspectRatioFilter|根据图像长宽比进行过滤|
|ImageResolutionFilter|根据图像分辨率进行过滤|