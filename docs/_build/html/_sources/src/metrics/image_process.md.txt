# Introduction to Image Data Processors
Note: We have set up corresponding filters for each scorer, named following the rules ```MetricnameScorer``` and ```MetricFilter```. Please refer to the [image_metrics.md](./image_metrics.md) for an introduction to these filters. The following introduces only the processors that differ from the previous scorers.


| Processor Name | Description |
|---|---|
| ImagePHashDeduplicator | Based on the [Perceptual hashing](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html) method from [imagededup](https://github.com/idealo/imagededup) |
| ImageDHashDeduplicator | Based on the [Difference hashing](https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html) method from [imagededup](https://github.com/idealo/imagededup) |
| ImageWHashDeduplicator | Based on the [Wavelet hashing](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5) method from [imagededup](https://github.com/idealo/imagededup) |
| ImageAHashDeduplicator | Based on the [Average hashing](https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html) method from [imagededup](https://github.com/idealo/imagededup) |
| ImageAspectRatioFilter | Filters images based on their aspect ratio |
| ImageResolutionFilter | Filters images based on their resolution |
