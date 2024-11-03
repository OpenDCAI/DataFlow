# 文本数据评估指标
## 概览
打分器分为以下四种类型，每种打分器会给出一个或多个分数。

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
      <td class="tg-0pky">APIcaller</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">调用API打分</td>
    </tr>
    <tr>
      <td class="tg-0pky">Diversity</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">计算整个数据集的多样性得分</td>
    </tr>
    <tr>
      <td class="tg-0pky">Models</td>
      <td class="tg-0pky">12</td>
      <td class="tg-0pky">基于模型、分类器打分</td>
    </tr>
    <tr>
      <td class="tg-0pky">Statistics</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">统计学指标打分</td>
    </tr>
  </tbody>
</table>

关于数据类型：【文本】表示接受单一字段字符串输入，可适用于预训练或微调数据。【指令】表示仅适用于微调数据多字段格式输入。

## 打分器列表

### APIcaller
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">评估维度</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">取值范围</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">AlpagasusScorer</td>
      <td class="tg-0pky">内容准确性与有效性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">通过调用 GPT 评估指令的质量，返回一个质量得分，得分越高表明指令的质量越高。</td>
      <td class="tg-0pky">[0, 5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2307.08701">paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PerspectiveScorer</td>
      <td class="tg-0pky">安全性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">使用 PerspectiveAPI 评估文本的毒性，返回毒性概率，得分越高表明文本毒性越高。</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky"><a href="https://perspectiveapi.com/">API</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">TreeinstructScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">通过生成语法树的节点数来衡量指令复杂性，节点越多表示指令越复杂。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2308.05696">paper</a></td>
    </tr>
  </tbody>
</table>

### Diversity
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">评估维度</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">取值范围</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">Task2VecScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">评估数据集的多样性，使用 Task2Vec 方法，高分表示数据集具有较高的多样性。</td>
      <td class="tg-0pky">[0.0525±3.41E-4, 0.4037±1.932E-5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2306.13840">paper</a><br><a href="https://github.com/alycialee/beyond-scale-language-data-diversity">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">VendiScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">通过计算 VendiScore 来评估数据集的多样性，得分越高表示多样性越高。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2210.02410">paper</a><br><a href="https://github.com/vertaix/Vendi-Score">code</a></td>
    </tr>
  </tbody>
</table>

### Models
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">评估维度</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">取值范围</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">DebertaV3Scorer</td>
      <td class="tg-0pky">内容准确性与有效性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">基于 Nvidia Deberta V3 模型的质量分类器，用于评估文本质量。</td>
      <td class="tg-0pky">{Low, Medium, High}</td>
      <td class="tg-0pky"><a href="https://huggingface.co/nvidia/quality-classifier-deberta">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">FineWebEduScorer</td>
      <td class="tg-0pky">教育价值</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">用于评估文本教育价值的分类器，高分表示文本具有较高的教育价值。</td>
      <td class="tg-0pky">[0, 5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2406.17557">paper</a><br><a href="https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">InstagScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">通过返回标签的数量来评估指令的内容多样性，标签越多表示内容多样性越大。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2308.07074">paper</a><br><a href="https://huggingface.co/OFA-Sys/InsTagger">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PerplexityScorer</td>
      <td class="tg-0pky">流畅性与可理解性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">基于 Kenlm 模型计算文本的困惑度，困惑度越低，文本的流畅性和可理解性越高。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://aclanthology.org/W11-2123.pdf">paper</a><br><a href="https://huggingface.co/edugp/kenlm/tree/main">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">QuratingScorer</td>
      <td class="tg-0pky">内容准确性与有效性、教育价值</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">通过 Qurating 模型评估文本的质量，得分越高表示质量越高。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2402.09739">paper</a><br><a href="https://github.com/princeton-nlp/QuRating">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PresidioScorer</td>
      <td class="tg-0pky">安全性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">使用Microsoft Presidio模型，识别文本中的私人实体（PII）如信用卡号、姓名、位置等。打分器返回PII信息个数。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://github.com/microsoft/presidio">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SuperfilteringScorer</td>
      <td class="tg-0pky">流畅性与可理解性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">使用 Superfiltering 方法评估指令的跟随难度，得分越高表示指令越难跟随。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2402.00530">paper</a><br><a href="https://github.com/tianyi-lab/Superfiltering">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">TextbookScorer</td>
      <td class="tg-0pky">教育价值</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">基于 FastText 分类器的课本质量分类器，用于评估文本的教育价值。</td>
      <td class="tg-0pky">[0, 2]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2306.11644">paper</a><br><a href="https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">UnievalScorer</td>
      <td class="tg-0pky">流畅性与可理解性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">UniEval多维度文本质量评估模型，得分越高表示质量越好。</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2210.07197">paper</a><br><a href="https://github.com/maszhongming/UniEval">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">DeitaQualityScorer</td>
      <td class="tg-0pky">内容准确性与有效性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">基于 Llama 模型的 Deita 指令质量评估器，高分表示指令质量较高。</td>
      <td class="tg-0pky">[1,6]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2312.15685">paper</a><br><a href="https://huggingface.co/hkust-nlp/deita-quality-scorer">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">DeitaComplexityScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">基于 Llama 模型的 Deita 指令复杂性评估器，高分表示指令复杂性较高。</td>
      <td class="tg-0pky">[1,6]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2312.15685">paper</a><br><a href="https://huggingface.co/hkust-nlp/deita-complexity-scorer">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RMScorer</td>
      <td class="tg-0pky">流畅性与可理解性</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">基于人类价值判断的奖励模型reward-model-deberta-v3-large-v2质量评分器。高分代表质量较高。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2">code</a></td>
    </tr>
  </tbody>
</table>

### Statistics
<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">名称</th>
      <th class="tg-0pky">评估维度</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">简介</th>
      <th class="tg-0pky">取值范围</th>
      <th class="tg-0pky">官方仓库或论文</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LangkitScorer</td>
      <td class="tg-0pky">文本结构, 流畅性与可理解性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">使用Langkit工具包计算文本的统计信息，如字数、句子数、音节数等，帮助评估文本的结构复杂性和可读性。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://github.com/whylabs/langkit">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">LexicalDiversityScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">使用MTLD和HDD方法计算词汇多样性评分，高分代表更丰富的词汇使用，反映文本的多样性和复杂性。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://link.springer.com/article/10.3758/BRM.42.2.381">paper</a><br><a href="https://github.com/jennafrens/lexical_diversity/tree/master">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">NgramScorer</td>
      <td class="tg-0pky">多样性与复杂性</td>
      <td class="tg-0pky">文本</td>
      <td class="tg-0pky">计算文本中n-gram的重复比例，用以衡量文本的重复度，得分越高表示文本中重复的n-gram比例越低。</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody>
</table>

## 质量评估体系

为提供更精准的数据质量评估，我们根据现有的分类器构架了一套质量评估体系。具体到每个打分器的输出分数指标，包括以下6个维度。

### 1. 文本结构 (Text Structure)
- **LangkitScorer**: LangkitSentenceCountScore, LangkitCharacterCountScore, LangkitLetterCountScore, LangkitSyllableCountScore, LangkitPolysyllableCountScore, LangkitMonosyllableCountScore, LangkitLexiconCountScore, LangkitDifficultWordsScore

### 2. 多样性与复杂性 (Diversity & Complexity)
- **LexicalDiversityScorer**: LexicalDiversityMTLDScore, LexicalDiversityHD-DScore
- **NgramScorer**: NgramScore
- **InstagScorer**: InstagScore
- **TreeinstructScorer**: TreeinstructScore
- **Task2VecScorer**: Task2VecDiversityScore (ConfidenceInterval)
- **VendiScorer**: N-gramsVendiScore, BERTVendiScore, SimCSEVendiScore
- **DeitaComplexityScorer:** DeitaComplexityScore


### 3. 流畅性与可理解性 (Fluency & Understandability)
- **UniEvalScorer**: UniEvalFluencyScore, UniEvalUnderstandabilityScore, UniEvalNaturalnessScore
- **LangkitScorer**: LangkitFleschReadingEaseScore, LangkitAutomatedReadabilityIndexScore, LangkitAggregateReadingLevelScore
- **PerplexityScorer**: PerplexityScore
- **QuratingScorer**: QuratingWritingStyleScore
- **SuperfilteringScorer**: SuperfilteringScore
- **RMScorer**: RMScore

### 4. 安全性 (Safety)
- **PerspectiveScorer**: PerspectiveScore
- **PresidioScorer**: PresidioScore

### 5. 教育价值 (Educational Value)
- **TextbookScorer**: TextbookScore
- **FineWebEduScorer**: FineWebEduScore
- **QuratingScorer**: QuratingEducationalValueScore

### 6. 内容准确性与有效性 (Content Accuracy & Effectiveness)
- **QuratingScorer**: QuratingRequiredExpertiseScore, QuratingFactsAndTriviaScore
- **DebertaV3Scorer**: DebertaV3Score
- **AlpagasusScorer**: AlpagasusScore
- **DeitaQualityScorer**: DeitaQualityScore

## 基准值

为更好的提供数据质量参考，我们根据数据类型从目前认为较高质量的[Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)和[alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)数据集中分别随机选取了5k条数据，并测试了部分打分器的基准值。

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">打分器名称</th>
    <th class="tg-0pky">分数指标名称</th>
    <th class="tg-0pky">简介</th>
    <th class="tg-0pky">均值</th>
    <th class="tg-0pky">方差</th>
    <th class="tg-0pky">最大值</th>
    <th class="tg-0pky">最小值</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="1">PerspectiveScorer</td>
    <td class="tg-0pky">PerspectiveScore</td>
    <td class="tg-0pky">评估文本的毒性，是否含有潜在的侮辱性或不当言论。<b>分数越高毒性越大。</b></td>
    <td class="tg-0pky">0.0426</td>
    <td class="tg-0pky">0.0025</td>
    <td class="tg-0pky">0.2610</td>
    <td class="tg-0pky">0.0026</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">LexicalDiversityScorer</td>
    <td class="tg-0pky">LexicalDiversityMTLDScore</td>
    <td class="tg-0pky">测量文本的词汇多样性。<b>分数越高词汇多样性越大。</b></td>
    <td class="tg-0pky">100.5990</td>
    <td class="tg-0pky">1625.1318</td>
    <td class="tg-0pky">1165.7164</td>
    <td class="tg-0pky">14.8439</td>
  </tr>
  <tr>
    <td class="tg-0pky">LexicalDiversityHD-DScore</td>
    <td class="tg-0pky">用于衡量文本的词汇多样性，基于离散分布计算。<b>分数越高词汇多样性越大。</b></td>
    <td class="tg-0pky">0.8487</td>
    <td class="tg-0pky">0.0014</td>
    <td class="tg-0pky">0.9873</td>
    <td class="tg-0pky">0.5570</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">NgramScorer</td>
    <td class="tg-0pky">NgramScore</td>
    <td class="tg-0pky">计算文本中n-gram的重复比例，用以衡量文本的重复度。<b>分数越高N-gram重复性越低。</b></td>
    <td class="tg-0pky">0.9938</td>
    <td class="tg-0pky">0.0002</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky">0.8285</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="11">LangkitScorer</td>
    <td class="tg-0pky">LangkitFleschReadingEaseScore</td>
    <td class="tg-0pky">衡量文本的Flesch可读性。<b>得分越高表示越易读。</b></td>
    <td class="tg-0pky">55.1870</td>
    <td class="tg-0pky">324.8975</td>
    <td class="tg-0pky">106.37</td>
    <td class="tg-0pky">-144.75</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitAutomatedReadabilityIndexScore</td>
    <td class="tg-0pky">自动可读性指标，基于句子长度和词汇难度。<b>得分越高表示越难读。</b></td>
    <td class="tg-0pky">11.7727</td>
    <td class="tg-0pky">19.4117</td>
    <td class="tg-0pky">98.2</td>
    <td class="tg-0pky">0.9</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitAggregateReadingLevelScore</td>
    <td class="tg-0pky">综合文本的阅读难度评分。<b>得分越高表示越难读。</b></td>
    <td class="tg-0pky">11.2332</td>
    <td class="tg-0pky">13.6816</td>
    <td class="tg-0pky">77.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitSyllableCountScore</td>
    <td class="tg-0pky">统计文本中音节的总数。<b>得分越高音节数量越大。</b></td>
    <td class="tg-0pky">815.3852</td>
    <td class="tg-0pky">2299853.7272</td>
    <td class="tg-0pky">43237</td>
    <td class="tg-0pky">32</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitLexiconCountScore</td>
    <td class="tg-0pky">统计文本中词汇的总数。<b>得分越高词汇数量越大。</b></td>
    <td class="tg-0pky">524.178</td>
    <td class="tg-0pky">1061058.5875</td>
    <td class="tg-0pky">33033</td>
    <td class="tg-0pky">23</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitSentenceCountScore</td>
    <td class="tg-0pky">统计文本中的句子数量。<b>得分越高句子数量越大。</b></td>
    <td class="tg-0pky">28.9664</td>
    <td class="tg-0pky">3618.2549</td>
    <td class="tg-0pky">2193</td>
    <td class="tg-0pky">1</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitCharacterCountScore</td>
    <td class="tg-0pky">统计文本中的字符数量。<b>得分越高字符数量越大。</b></td>
    <td class="tg-0pky">2610.2462</td>
    <td class="tg-0pky">23580442.8820</td>
    <td class="tg-0pky">139807</td>
    <td class="tg-0pky">118</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitLetterCountScore</td>
    <td class="tg-0pky">统计文本中的字母数量。<b>得分越高字母数量越大。</b></td>
    <td class="tg-0pky">2513.4572</td>
    <td class="tg-0pky">21890120.2030</td>
    <td class="tg-0pky">134507</td>
    <td class="tg-0pky">109</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitPolysyllableCountScore</td>
    <td class="tg-0pky">统计多音节单词的数量。<b>得分越高多音节词数量越大。</b></td>
    <td class="tg-0pky">78.8834</td>
    <td class="tg-0pky">18918.1990</td>
    <td class="tg-0pky">3261</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitMonosyllableCountScore</td>
    <td class="tg-0pky">统计单音节单词的数量，通常与文本的简易度相关。<b>得分越高单音节词数量越大。</b></td>
    <td class="tg-0pky">334.6674</td>
    <td class="tg-0pky">503285.5160</td>
    <td class="tg-0pky">25133</td>
    <td class="tg-0pky">13</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitDifficultWordsScore</td>
    <td class="tg-0pky">统计文本中难词的数量。<b>得分越高难词数量越大。</b></td>
    <td class="tg-0pky">93.4112</td>
    <td class="tg-0pky">14401.2789</td>
    <td class="tg-0pky">2366</td>
    <td class="tg-0pky">4</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">UnievalScorer</td>
    <td class="tg-0pky">UniEvalFluencyScore</td>
    <td class="tg-0pky">评估文本的流畅性。<b>得分越高文本越流畅。</b></td>
    <td class="tg-0pky">0.8268</td>
    <td class="tg-0pky">0.0199</td>
    <td class="tg-0pky">0.9674</td>
    <td class="tg-0pky">0.0036</td>
  </tr>
  <tr>
    <td class="tg-0pky">UniEvalNaturalnessScore</td>
    <td class="tg-0pky">测量文本的自然性。<b>得分越高文本越像自然语言。</b></td>
    <td class="tg-0pky">0.4224</td>
    <td class="tg-0pky">0.0474</td>
    <td class="tg-0pky">0.9782</td>
    <td class="tg-0pky">0.0010</td>
  </tr>
  <tr>
    <td class="tg-0pky">UniEvalUnderstandabilityScore</td>
    <td class="tg-0pky">衡量文本的可理解性。<b>得分越高文本越可理解。</b></td>
    <td class="tg-0pky">0.4698</td>
    <td class="tg-0pky">0.0493</td>
    <td class="tg-0pky">0.9927</td>
    <td class="tg-0pky">0.0006</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">TextbookScorer</td>
    <td class="tg-0pky">TextbookScore</td>
    <td class="tg-0pky">测试文本是否符合教科书标准。<b>得分越高文本越接近理想教材。</b></td>
    <td class="tg-0pky">0.9255</td>
    <td class="tg-0pky">0.1779</td>
    <td class="tg-0pky">1.9867</td>
    <td class="tg-0pky">0.0001</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">FineWebEduScorer</td>
    <td class="tg-0pky">FineWebEduScore</td>
    <td class="tg-0pky">测量文本的教育价值。<b>得分越高文本教育价值越大。</b></td>
    <td class="tg-0pky">1.1901</td>
    <td class="tg-0pky">0.4924</td>
    <td class="tg-0pky">4.6827</td>
    <td class="tg-0pky">-0.6319</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">DebertaV3Scorer</td>
    <td class="tg-0pky">DebertaV3Score</td>
    <td class="tg-0pky">使用DebertaV3模型进行的文本评估。<b>评估质量得分按高、中、低分类。</b></td>
    <td class="tg-0pky">Medium: 3180 次</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">High: 1412 次</td>
    <td class="tg-0pky">Low: 408 次</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">PerplexityScorer</td>
    <td class="tg-0pky">PerplexityScore</td>
    <td class="tg-0pky">衡量文本的困惑度。<b>得分越高模型困惑度越大。</b></td>
    <td class="tg-0pky">564.3942</td>
    <td class="tg-0pky">165893.5542</td>
    <td class="tg-0pky">8271.0</td>
    <td class="tg-0pky">13.9</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">QuratingScorer</td>
    <td class="tg-0pky">QuratingWritingStyleScore</td>
    <td class="tg-0pky">评估文本的写作风格是否良好。<b>得分越高文本写作风格越好。</b></td>
    <td class="tg-0pky">0.6453</td>
    <td class="tg-0pky">6.7949</td>
    <td class="tg-0pky">8.375</td>
    <td class="tg-0pky">-7.3474</td>
  </tr>
  <tr>
    <td class="tg-0pky">QuratingRequiredExpertiseScore</td>
    <td class="tg-0pky">衡量文本需要的专业知识水平。<b>得分越高文本越需要专业知识。</b></td>
    <td class="tg-0pky">-0.4661</td>
    <td class="tg-0pky">7.0458</td>
    <td class="tg-0pky">9.0</td>
    <td class="tg-0pky">-8.25</td>
  </tr>
  <tr>
    <td class="tg-0pky">QuratingFactsAndTriviaScore</td>
    <td class="tg-0pky">测试文本是否包含事实和趣闻。<b>得分越高文本包含的事实和趣闻越多。</b></td>
    <td class="tg-0pky">0.1889</td>
    <td class="tg-0pky">4.5678</td>
    <td class="tg-0pky">7.4688</td>
    <td class="tg-0pky">-6.0993</td>
  </tr>
  <tr>
    <td class="tg-0pky">QuratingEducationalValueScore</td>
    <td class="tg-0pky">衡量文本的教育价值。<b>得分越高文本教育价值越大。</b></td>
    <td class="tg-0pky">1.2946</td>
    <td class="tg-0pky">11.2196</td>
    <td class="tg-0pky">11.5625</td>
    <td class="tg-0pky">-8.7843</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">InstagScorer</td>
    <td class="tg-0pky">InstagScore</td>
    <td class="tg-0pky">通过返回标签的数量来评估指令的内容多样性。<b>得分越高内容多样性越大。</b></td>
    <td class="tg-0pky">2.304</td>
    <td class="tg-0pky">2.9396</td>
    <td class="tg-0pky">11</td>
    <td class="tg-0pky">1</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">SuperfilteringScorer</td>
    <td class="tg-0pky">SuperfilteringScore</td>
    <td class="tg-0pky">使用 Superfiltering 方法评估指令的跟随难度。<b>得分越高指令跟随难度越大。</b></td>
    <td class="tg-0pky">1.3223</td>
    <td class="tg-0pky">836.0302</td>
    <td class="tg-0pky">1978.6534</td>
    <td class="tg-0pky">0.0011</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">DeitaQualityScorer</td>
    <td class="tg-0pky">DeitaQualityScore</td>
    <td class="tg-0pky">基于 Llama 模型的 Deita 指令质量评估器。<b>得分越高指令质量越好。</b></td>
    <td class="tg-0pky">3.5629</td>
    <td class="tg-0pky">0.9247</td>
    <td class="tg-0pky">5.5309</td>
    <td class="tg-0pky">1.0840</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">DeitaComplexityScorer</td>
    <td class="tg-0pky">DeitaComplexityScore</td>
    <td class="tg-0pky">基于 Llama 模型的 Deita 指令复杂性评估器。<b>得分越高指令复杂性越大。</b></td>
    <td class="tg-0pky">1.4936</td>
    <td class="tg-0pky">0.2086</td>
    <td class="tg-0pky">3.3207</td>
    <td class="tg-0pky">1.0001</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">VendiScorer</td>
    <td class="tg-0pky">N-grams_VendiScore</td>
    <td class="tg-0pky">基于N-grams嵌入评估文本多样性得分。<b>得分越高数据集多样性越大。</b></td>
    <td class="tg-0pky">1832.96</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT_VendiScore</td>
    <td class="tg-0pky">基于BERT嵌入评估文本多样性得分。<b>得分越高数据集多样性越大。</b></td>
    <td class="tg-0pky">1.83</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">SimCSE_VendiScore</td>
    <td class="tg-0pky">基于SimCSE嵌入计算文本多样性得分。<b>得分越高数据集多样性越大。</b></td>
    <td class="tg-0pky">68.94</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">Task2VecScorer</td>
    <td class="tg-0pky">Task2VecScore</td>
    <td class="tg-0pky">使用Task2Vec多样性系数评估数据集多样性。<b>得分越高数据集多样性越大。</b></td>
    <td class="tg-0pky">0.0673</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">AlpagasusScorer</td>
    <td class="tg-0pky">AlpagasusScore</td>
    <td class="tg-0pky">调用ChatGPT评估指令质量得分。<b>得分越高指令质量越好。</b></td>
    <td class="tg-0pky">4.172</td>
    <td class="tg-0pky">0.2164</td>
    <td class="tg-0pky">5.0</td>
    <td class="tg-0pky">2.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">TreeinstructScorer</td>
    <td class="tg-0pky">TreeinstructScore</td>
    <td class="tg-0pky">调用ChatGPT评估指令语义复杂度。<b>得分越高指令语义复杂度越高。</b></td>
    <td class="tg-0pky">6.494</td>
    <td class="tg-0pky">9.7540</td>
    <td class="tg-0pky">63.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">PresidioScorer</td>
    <td class="tg-0pky">PresidioScore</td>
    <td class="tg-0pky">使用Presidio评估PII个数。<b>得分越高文本含义PII信息越多。</b></td>
    <td class="tg-0pky">21.4008</td>
    <td class="tg-0pky">2915.3542</td>
    <td class="tg-0pky">1786.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">RMScorer</td>
    <td class="tg-0pky">RMScore</td>
    <td class="tg-0pky">使用基于人类价值的奖励模型评估SFT数据质量<b>得分越高数据质量越高。</b></td>
    <td class="tg-0pky">3.1537</td>
    <td class="tg-0pky">9.9461</td>
    <td class="tg-0pky">8.6803</td>
    <td class="tg-0pky">-4.9680</td>
  </tr>
</tbody>
</table>
