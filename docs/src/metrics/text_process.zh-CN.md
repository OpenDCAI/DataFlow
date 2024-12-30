# 文本数据处理
## 概览
DataFlow目前支持的文本数据处理主要针对于数据点层面，可以分为以下三种类型，分别是数据改写器、数据去重器和数据过滤器。
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">类型</th>
      <th class="tg-0pky">数量</th>
      <th class="tg-0pky">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">数据改写器</td>
      <td class="tg-0pky">15</td>
      <td class="tg-0pky">通过数据处理、数据增强等方式改善数据点内容（不改变总数量）</td>
    </tr>
    <tr>
      <td class="tg-0pky">数据去重器</td>
      <td class="tg-0pky">6</td>
      <td class="tg-0pky">通过哈希等方法进行数据点去重</td>
    </tr>
    <tr>
      <td class="tg-0pky">数据过滤器</td>
      <td class="tg-0pky">42</td>
      <td class="tg-0pky">通过设置阈值等方式过滤数据点</td>
    </tr>
  </tbody>
</table>

## 数据改写器

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">适用类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LowercaseRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">将文本字段中的内容转换为小写</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">PIIAnonymizeRefiner</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">通过识别和匿名化个人身份信息（PII），如姓名、位置等，来保护隐私</td>
      <td class="tg-0pky"><a href="https://github.com/microsoft/presidio">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemovePunctuationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">移除文本中的标点符号</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveNumberRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">移除文本中的数字字符</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveExtraSpacesRefiner</td>
      <td class="tg-0pky">NLP、预训练</td>
      <td class="tg-0pky">移除文本中的多余空格，将连续的多个空格替换为单个空格，并去除文本前后空格</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveRepetitionsPunctuationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">移除重复的标点符号，例如“!!!”变为“!”</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveEmojiRefiner</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">移除文本中的表情符号，例如"😀"</td>
      <td class="tg-0pky"><a href="https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveEmoticonsRefiner</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">移除文本中的表情符号，例如“:‑)”，使用预定义的表情符号列表</td>
      <td class="tg-0pky"><a href="https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveContractionsRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">扩展文本中的缩写词（例如将“can't”扩展为“cannot”）</td>
      <td class="tg-0pky"><a href="https://github.com/kootenpv/contractions">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">HtmlUrlRemoverRefiner</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">移除文本中的URL和HTML标签</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">TextNormalizationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">规范化文本中的日期格式、货币格式等</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">NERRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">使用命名实体识别（NER）技术识别并屏蔽文本中的特定实体</td>
      <td class="tg-0pky"><a href="https://spacy.io/usage/linguistic-features#named-entities">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">StemmingLemmatizationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">对文本进行词干提取或词形还原</td>
      <td class="tg-0pky"><a href="https://www.nltk.org/">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SpellingCorrectionRefiner</td>
      <td class="tg-0pky">NLP、预训练</td>
      <td class="tg-0pky">通过SymSpell对文本中的拼写错误进行纠正</td>
      <td class="tg-0pky"><a href="https://github.com/mammothb/symspellpy">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveStopwordsRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">移除文本中的停用词（如“the”，“is”）</td>
      <td class="tg-0pky"><a href="https://github.com/nltk/nltk">Code</a></td>
    </tr>
  </tbody>
</table>


## 数据去重器
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">适用类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">HashDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">使用多种哈希函数（如MD5、SHA256、XXH3_128）对文本进行哈希处理，通过精确的比较哈希值来识别和移除重复数据，适用于小规模简单去重场景。</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">CCNetDeduplicator</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">基于SHA-1哈希算法的前64位进行比较，以识别重复文本。旨在平衡哈希安全性和计算效率。</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">NgramHashDeduplicator</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">结合n-gram技术与哈希算法，将文本分割为多个n-gram片段并分别进行哈希处理。通过多个哈希值的比较来识别相似或重复的文本，适用于处理具有细微差异的重复数据。</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1607.04606">Paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SemDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">基于BERT模型的语义相似度计算，通过生成文本的嵌入向量并计算余弦相似度来识别重复内容。适用于需要语义理解的高级去重场景，能够识别语义上相似但表述不同的文本。</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1810.04805">Paper</a> <br> <a href="https://github.com/facebookresearch/SemDeDup">Code</a> </td>
    </tr>
    <tr>
      <td class="tg-0pky">SimHashDeduplicator</td>
      <td class="tg-0pky">预训练</td>
      <td class="tg-0pky">采用SimHash算法，通过生成文本的SimHash指纹并计算汉明距离来判断文本的相似度。适用于高效的相似文本检测，能够快速处理大规模数据集中的重复或相似文本。</td>
      <td class="tg-0pky"><a href="https://dl.acm.org/doi/abs/10.1145/1242572.1242592">Paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">MinHashDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">结合MinHash与LSH，通过将集合中的元素哈希成一个较小的签名（通常是一个固定长度的整数或比特串），从而以很小的内存占用和低计算成本比较两个集合之间的相似度。</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1811.04633">Paper</a></td>
    </tr>
  </tbody>
</table>

## 数据过滤器

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">适用类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LanguageFilter</td>
      <td class="tg-0pky">预训练、SFT</td>
      <td class="tg-0pky">使用fasttext语言识别模型过滤特定语言</td>
      <td class="tg-0pky"><a href="https://huggingface.co/facebook/fasttext-language-identification">Huggingface</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">BlocklistFilter</td>
      <td class="tg-0pky">预训练、SFT</td>
      <td class="tg-0pky">设置阈值，根据List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words屏蔽词表过滤数据点</td>
      <td class="tg-0pky"><a href="https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words?tab=readme-ov-file">Code</a></td>
    </tr>
  </tbody>
</table>

同时，作为配套工作，我们支持了Open-DataFlow-Eval文本数据评估模块中基于单个数据点打分器评分的过滤。共18种。
```yaml
DeitaQualityFilter:
    min_score: 1                                         
    max_score: 5                                     
    scorer_args:
      device: 'cuda:0'
      model_name: 'hkust-nlp/deita-quality-scorer'
      max_length: 512
```
可通过设置需要保留的`min/max`分数并在`scorer_args`中设置打分器参数实现。

支持的打分器，详见[评估算法文档](text_metrics.zh-CN.md)（除Diversity部分）。

此外，启发式规则过滤在预训练数据的筛选方面占有很大的比重，在这一方面，[Dingo数据质量评估工具](https://github.com/DataEval/dingo)对我们的开发带来了很大的启发。我们在`dataflow/process/text/filters/heuristics.py`中整合了部分Dingo中使用的规则过滤算法，共22种。详见[规则文档](https://github.com/DataEval/dingo/blob/dev/docs/rules.md)，过滤器名称可参考`dataflow/process/text/filters/heuristics.py`文件。

需要说明的是，以上提到的42种数据过滤器具有相同的`Yaml`调用方式。
