# Text Data Evaluation Metrics

## Overview

Scorers are divided into the following four types, each scorer provides one or more scores.

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Type</th>
      <th class="tg-0pky">Count</th>
      <th class="tg-0pky">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">APIcaller</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">Call API for scoring</td>
    </tr>
    <tr>
      <td class="tg-0pky">Diversity</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">Compute diversity score of the entire dataset</td>
    </tr>
    <tr>
      <td class="tg-0pky">Models</td>
      <td class="tg-0pky">12</td>
      <td class="tg-0pky">Model or classifier-based scoring</td>
    </tr>
    <tr>
      <td class="tg-0pky">Statistics</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">Statistical metric scoring</td>
    </tr>
  </tbody>
</table>

Regarding data types: **[Text]** indicates accepting single-field string input, suitable for pre-training or fine-tuning data. **[Instruction]** indicates only suitable for fine-tuning data with multi-field format input.

## List of Scorers

### APIcaller

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Evaluation Dimension</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Value Range</th>
      <th class="tg-0pky">Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">AlpagasusScorer</td>
      <td class="tg-0pky">Content Accuracy & Effectiveness</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">Evaluates the quality of instructions by calling GPT, returning a quality score. A higher score indicates higher instruction quality.</td>
      <td class="tg-0pky">[0, 5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2307.08701">paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PerspectiveScorer</td>
      <td class="tg-0pky">Safety</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Uses PerspectiveAPI to evaluate the toxicity of the text, returning a toxicity probability. A higher score indicates higher text toxicity.</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky"><a href="https://perspectiveapi.com/">API</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">TreeinstructScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">Measures instruction complexity by generating the number of nodes in the syntax tree; more nodes indicate more complex instructions.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2308.05696">paper</a></td>
    </tr>
  </tbody>
</table>

### Diversity

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Evaluation Dimension</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Value Range</th>
      <th class="tg-0pky">Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">Task2VecScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Evaluates the diversity of the dataset using the Task2Vec method. Higher scores indicate higher dataset diversity.</td>
      <td class="tg-0pky">[0.0525±3.41E-4, 0.4037±1.932E-5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2306.13840">paper</a><br><a href="https://github.com/alycialee/beyond-scale-language-data-diversity">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">VendiScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Evaluates dataset diversity by calculating VendiScore; higher scores indicate higher diversity.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2210.02410">paper</a><br><a href="https://github.com/vertaix/Vendi-Score">code</a></td>
    </tr>
  </tbody>
</table>

### Models

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Evaluation Dimension</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Value Range</th>
      <th class="tg-0pky">Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">DebertaV3Scorer</td>
      <td class="tg-0pky">Content Accuracy & Effectiveness</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">A quality classifier based on NVIDIA's DeBERTa V3 model for evaluating text quality.</td>
      <td class="tg-0pky">{Low, Medium, High}</td>
      <td class="tg-0pky"><a href="https://huggingface.co/nvidia/quality-classifier-deberta">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">FineWebEduScorer</td>
      <td class="tg-0pky">Educational Value</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">A classifier for evaluating the educational value of text; higher scores indicate higher educational value.</td>
      <td class="tg-0pky">[0, 5]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2406.17557">paper</a><br><a href="https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">InstagScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">Evaluates instruction content diversity by returning the number of tags; more tags indicate higher content diversity.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2308.07074">paper</a><br><a href="https://huggingface.co/OFA-Sys/InsTagger">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PerplexityScorer</td>
      <td class="tg-0pky">Fluency & Understandability</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Calculates text perplexity using the KenLM model; lower scores indicate higher fluency and understandability.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://aclanthology.org/W11-2123.pdf">paper</a><br><a href="https://huggingface.co/edugp/kenlm/tree/main">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">QuratingScorer</td>
      <td class="tg-0pky">Content Accuracy & Effectiveness、 Educational Value</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Evaluates text quality using the Qurating model; higher scores indicate higher quality.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2402.09739">paper</a><br><a href="https://github.com/princeton-nlp/QuRating">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">PresidioScorer</td>
      <td class="tg-0pky">Safety</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Using the Microsoft Presidio model, identify private entities (PII) in text such as credit card numbers, names, locations, etc. The scorer returns the number of PII information.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://github.com/microsoft/presidio">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SuperfilteringScorer</td>
      <td class="tg-0pky">Fluency & Understandability</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">Evaluates the following difficulty of instructions using the Superfiltering method; higher scores indicate more difficult instructions to follow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2402.00530">paper</a><br><a href="https://github.com/tianyi-lab/Superfiltering">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">TextbookScorer</td>
      <td class="tg-0pky">Educational Value</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">A textbook quality classifier based on FastText, used to evaluate the educational value of text.</td>
      <td class="tg-0pky">[0, 2]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2306.11644">paper</a><br><a href="https://huggingface.co/kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">UnievalScorer</td>
      <td class="tg-0pky">Fluency & Understandability</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">UniEval is a multi-dimensional text quality evaluation model; higher scores indicate better quality.</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2210.07197">paper</a><br><a href="https://github.com/maszhongming/UniEval">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">DeitaQualityScorer</td>
      <td class="tg-0pky">Content Accuracy & Effectiveness</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">An instruction quality scorer based on the Llama model; higher scores indicate higher instruction quality.</td>
      <td class="tg-0pky">[1, 6]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2312.15685">paper</a><br><a href="https://huggingface.co/hkust-nlp/deita-quality-scorer">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">DeitaComplexityScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Instruction</td>
      <td class="tg-0pky">An instruction complexity scorer based on the Llama model; higher scores indicate higher instruction complexity.</td>
      <td class="tg-0pky">[1,6]</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/2312.15685">paper</a><br><a href="https://huggingface.co/hkust-nlp/deita-complexity-scorer">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RMScorer</td>
      <td class="tg-0pky">Fluency & Understandability</td>
      <td class="tg-0pky">指令</td>
      <td class="tg-0pky">A reward-model-deberta-v3-large-v2 scorer based on human value judgment. High scores represent higher quality.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2">code</a></td>
    </tr>
  </tbody>
</table>

### Statistics

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Evaluation Dimension</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Value Range</th>
      <th class="tg-0pky">Official Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LangkitScorer</td>
      <td class="tg-0pky">Text Structure, Fluency & Understandability</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Calculates statistical information of text using the Langkit toolkit, such as word count, sentence count, syllable count, etc., to help evaluate the structural complexity and readability of the text.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://github.com/whylabs/langkit">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">LexicalDiversityScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Calculates lexical diversity scores using MTLD and HD-D methods; higher scores represent richer vocabulary use, reflecting the diversity and complexity of the text.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky"><a href="https://link.springer.com/article/10.3758/BRM.42.2.381">paper</a><br><a href="https://github.com/jennafrens/lexical_diversity/tree/master">code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">NgramScorer</td>
      <td class="tg-0pky">Diversity & Complexity</td>
      <td class="tg-0pky">Text</td>
      <td class="tg-0pky">Calculates the repetition ratio of n-grams in the text to measure text repetition; higher scores indicate lower repetition of n-grams in the text.</td>
      <td class="tg-0pky">[0, 1]</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody>
</table>

## Quality Evaluation System

To provide more precise data quality evaluation, we have constructed a quality evaluation system based on existing classifiers. Specifically, the output score metrics of each scorer include the following six dimensions.

### 1. Text Structure

- **LangkitScorer**: LangkitSentenceCountScore, LangkitCharacterCountScore, LangkitLetterCountScore, LangkitSyllableCountScore, LangkitPolysyllableCountScore, LangkitMonosyllableCountScore, LangkitLexiconCountScore, LangkitDifficultWordsScore

### 2. Diversity & Complexity

- **LexicalDiversityScorer**: LexicalDiversityMTLDScore, LexicalDiversityHD-DScore
- **NgramScorer**: NgramScore
- **InstagScorer**: InstagScore
- **TreeinstructScorer**: TreeinstructScore
- **Task2VecScorer**: Task2VecDiversityScore (ConfidenceInterval)
- **VendiScorer**: N-gramsVendiScore, BERTVendiScore, SimCSEVendiScore
- **DeitaComplexityScorer:** DeitaComplexityScore

### 3. Fluency & Understandability

- **UniEvalScorer**: UniEvalFluencyScore, UniEvalUnderstandabilityScore, UniEvalNaturalnessScore
- **LangkitScorer**: LangkitFleschReadingEaseScore, LangkitAutomatedReadabilityIndexScore, LangkitAggregateReadingLevelScore
- **PerplexityScorer**: PerplexityScore
- **QuratingScorer**: QuratingWritingStyleScore
- **SuperfilteringScorer**: SuperfilteringScore
- **RMScorer**: RMScore

### 4. Safety

- **PerspectiveScorer**: PerspectiveScore
- **PresidioScorer**: PresidioScore

### 5. Educational Value

- **TextbookScorer**: TextbookScore
- **FineWebEduScorer**: FineWebEduScore
- **QuratingScorer**: QuratingEducationalValueScore

### 6. Content Accuracy & Effectiveness

- **QuratingScorer**: QuratingRequiredExpertiseScore, QuratingFactsAndTriviaScore
- **DebertaV3Scorer**: DebertaV3Score
- **AlpagasusScorer**: AlpagasusScore
- **DeitaScorer**: DeitaScore

## Benchmark Values

To better provide data quality references, we randomly selected 5k data samples from the currently considered high-quality datasets [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) based on data types, and tested the benchmark values of some scorers.

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Scorer Name</th>
    <th class="tg-0pky">Score Metric Name</th>
    <th class="tg-0pky">Description</th>
    <th class="tg-0pky">Mean</th>
    <th class="tg-0pky">Variance</th>
    <th class="tg-0pky">Max</th>
    <th class="tg-0pky">Min</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="1">PerspectiveScorer</td>
    <td class="tg-0pky">PerspectiveScore</td>
    <td class="tg-0pky">Evaluates the toxicity of the text, checking for potential insults or inappropriate language. <b>The higher the score, the higher the toxicity</b></td>
    <td class="tg-0pky">0.0426</td>
    <td class="tg-0pky">0.0025</td>
    <td class="tg-0pky">0.2610</td>
    <td class="tg-0pky">0.0026</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">LexicalDiversityScorer</td>
    <td class="tg-0pky">LexicalDiversityMTLDScore</td>
    <td class="tg-0pky">Measures the lexical diversity of the text; higher scores indicate more varied vocabulary usage.<b>The higher the score, the higher the lexical diversity</b></td>
    <td class="tg-0pky">100.5990</td>
    <td class="tg-0pky">1625.1318</td>
    <td class="tg-0pky">1165.7164</td>
    <td class="tg-0pky">14.8439</td>
  </tr>
  <tr>
    <td class="tg-0pky">LexicalDiversityHD-DScore</td>
    <td class="tg-0pky">Used to measure the lexical diversity of the text, calculated based on discrete distribution.<b>The higher the score, the higher the lexical diversity</b></td>
    <td class="tg-0pky">0.8487</td>
    <td class="tg-0pky">0.0014</td>
    <td class="tg-0pky">0.9873</td>
    <td class="tg-0pky">0.5570</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">NgramScorer</td>
    <td class="tg-0pky">NgramScore</td>
    <td class="tg-0pky">Calculate the repetition ratio of n-grams in the text to measure the degree of repetition. <b>The higher the score, the lower the n-gram repetition.</b></td>
    <td class="tg-0pky">0.9938</td>
    <td class="tg-0pky">0.0002</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky">0.8285</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="11">LangkitScorer</td>
    <td class="tg-0pky">LangkitFleschReadingEaseScore</td>
    <td class="tg-0pky">Measures Flesch text readability. <b>The higher the score, the easier readability.</b></td>
    <td class="tg-0pky">55.1870</td>
    <td class="tg-0pky">324.8975</td>
    <td class="tg-0pky">106.37</td>
    <td class="tg-0pky">-144.75</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitAutomatedReadabilityIndexScore</td>
    <td class="tg-0pky">Automated readability index based on sentence length and vocabulary difficulty.<b>The higher the score, the more difficult readability</b></td>
    <td class="tg-0pky">11.7727</td>
    <td class="tg-0pky">19.4117</td>
    <td class="tg-0pky">98.2</td>
    <td class="tg-0pky">0.9</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitAggregateReadingLevelScore</td>
    <td class="tg-0pky">Aggregate reading difficulty score of the text.<b>The higher the score, the more difficult readability</b></td>
    <td class="tg-0pky">11.2332</td>
    <td class="tg-0pky">13.6816</td>
    <td class="tg-0pky">77.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitSyllableCountScore</td>
    <td class="tg-0pky">Counts the total number of syllables in the text. <b>The higher the score, the more syllables there are.</b></td>
    <td class="tg-0pky">815.3852</td>
    <td class="tg-0pky">2299853.7272</td>
    <td class="tg-0pky">43237</td>
    <td class="tg-0pky">32</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitLexiconCountScore</td>
    <td class="tg-0pky">Counts the total number of words in the text. <b>The higher the score, the more words there are.</b></td>
    <td class="tg-0pky">524.178</td>
    <td class="tg-0pky">1061058.5875</td>
    <td class="tg-0pky">33033</td>
    <td class="tg-0pky">23</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitSentenceCountScore</td>
    <td class="tg-0pky">Counts the total number of sentences in the text. <b>The higher the score, the more sentences there are.</b></td>
    <td class="tg-0pky">28.9664</td>
    <td class="tg-0pky">3618.2549</td>
    <td class="tg-0pky">2193</td>
    <td class="tg-0pky">1</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitCharacterCountScore</td>
    <td class="tg-0pky">Counts the total number of characters in the text. <b>The higher the score, the more characters there are.</b></td>
    <td class="tg-0pky">2610.2462</td>
    <td class="tg-0pky">23580442.8820</td>
    <td class="tg-0pky">139807</td>
    <td class="tg-0pky">118</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitLetterCountScore</td>
    <td class="tg-0pky">Counts the total number of letters in the text. <b>The higher the score, the more letters there are.</b></td>
    <td class="tg-0pky">2513.4572</td>
    <td class="tg-0pky">21890120.2030</td>
    <td class="tg-0pky">134507</td>
    <td class="tg-0pky">109</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitPolysyllableCountScore</td>
    <td class="tg-0pky">Counts the number of polysyllabic words in the text. <b>The higher the score, the more polysyllabic words there are.</b></td>
    <td class="tg-0pky">78.8834</td>
    <td class="tg-0pky">18918.1990</td>
    <td class="tg-0pky">3261</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitMonosyllableCountScore</td>
    <td class="tg-0pky">Counts the number of monosyllabic words, which are usually related to the text's simplicity. <b>The higher the score, the more monosyllabic words there are.</b></td>
    <td class="tg-0pky">334.6674</td>
    <td class="tg-0pky">503285.5160</td>
    <td class="tg-0pky">25133</td>
    <td class="tg-0pky">13</td>
  </tr>
  <tr>
    <td class="tg-0pky">LangkitDifficultWordsScore</td>
    <td class="tg-0pky">Counts the number of difficult words in the text. <b>The higher the score, the more difficult words there are.</b></td>
    <td class="tg-0pky">93.4112</td>
    <td class="tg-0pky">14401.2789</td>
    <td class="tg-0pky">2366</td>
    <td class="tg-0pky">4</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">UnievalScorer</td>
    <td class="tg-0pky">UniEvalFluencyScore</td>
    <td class="tg-0pky">Evaluates the fluency of the text. <b>The higher the score, the more fluent the text is.</b></td>
    <td class="tg-0pky">0.8268</td>
    <td class="tg-0pky">0.0199</td>
    <td class="tg-0pky">0.9674</td>
    <td class="tg-0pky">0.0036</td>
  </tr>
  <tr>
      <td class="tg-0pky">UniEvalNaturalnessScore</td>
      <td class="tg-0pky">Measures the naturalness of the text. <b>The higher the score, the more natural the language.</b></td>
      <td class="tg-0pky">0.4224</td>
      <td class="tg-0pky">0.0474</td>
      <td class="tg-0pky">0.9782</td>
      <td class="tg-0pky">0.0010</td>
  </tr>
  <tr>
      <td class="tg-0pky">UniEvalUnderstandabilityScore</td>
      <td class="tg-0pky">Measures the understandability of the text. <b>The higher the score, the more understandable the text.</b></td>
      <td class="tg-0pky">0.4698</td>
      <td class="tg-0pky">0.0493</td>
      <td class="tg-0pky">0.9927</td>
      <td class="tg-0pky">0.0006</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">TextbookScorer</td>
      <td class="tg-0pky">TextbookScore</td>
      <td class="tg-0pky">Tests whether the text meets textbook standards. <b>The higher the score, the closer the text is to an ideal textbook.</b></td>
      <td class="tg-0pky">0.9255</td>
      <td class="tg-0pky">0.1779</td>
      <td class="tg-0pky">1.9867</td>
      <td class="tg-0pky">0.0001</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">FineWebEduScorer</td>
      <td class="tg-0pky">FineWebEduScore</td>
      <td class="tg-0pky">Measures the educational value of the text. <b>The higher the score, the greater the educational value.</b></td>
      <td class="tg-0pky">1.1901</td>
      <td class="tg-0pky">0.4924</td>
      <td class="tg-0pky">4.6827</td>
      <td class="tg-0pky">-0.6319</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">DebertaV3Scorer</td>
      <td class="tg-0pky">DebertaV3Score</td>
      <td class="tg-0pky">Text evaluation using the DebertaV3 model. <b>Quality scores are classified as high, medium, or low.</b></td>
      <td class="tg-0pky">Medium: 3180 times</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">High: 1412 times</td>
      <td class="tg-0pky">Low: 408 times</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">PerplexityScorer</td>
      <td class="tg-0pky">PerplexityScore</td>
      <td class="tg-0pky">Measures the perplexity of the text. <b>The higher the score, the greater the model's perplexity.</b></td>
      <td class="tg-0pky">564.3942</td>
      <td class="tg-0pky">165893.5542</td>
      <td class="tg-0pky">8271.0</td>
      <td class="tg-0pky">13.9</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="4">QuratingScorer</td>
      <td class="tg-0pky">QuratingWritingStyleScore</td>
      <td class="tg-0pky">Evaluates the quality of the text's writing style. <b>The higher the score, the better the writing style.</b></td>
      <td class="tg-0pky">0.6453</td>
      <td class="tg-0pky">6.7949</td>
      <td class="tg-0pky">8.375</td>
      <td class="tg-0pky">-7.3474</td>
  </tr>
  <tr>
      <td class="tg-0pky">QuratingRequiredExpertiseScore</td>
      <td class="tg-0pky">Measures the level of expertise required for the text. <b>The higher the score, the more expertise is required.</b></td>
      <td class="tg-0pky">-0.4661</td>
      <td class="tg-0pky">7.0458</td>
      <td class="tg-0pky">9.0</td>
      <td class="tg-0pky">-8.25</td>
  </tr>
  <tr>
      <td class="tg-0pky">QuratingFactsAndTriviaScore</td>
      <td class="tg-0pky">Tests whether the text contains facts and trivia. <b>The higher the score, the more facts and trivia the text contains.</b></td>
      <td class="tg-0pky">0.1889</td>
      <td class="tg-0pky">4.5678</td>
      <td class="tg-0pky">7.4688</td>
      <td class="tg-0pky">-6.0993</td>
  </tr>
  <tr>
      <td class="tg-0pky">QuratingEducationalValueScore</td>
      <td class="tg-0pky">Measures the educational value of the text. <b>The higher the score, the greater the educational value.</b></td>
      <td class="tg-0pky">1.2946</td>
      <td class="tg-0pky">11.2196</td>
      <td class="tg-0pky">11.5625</td>
      <td class="tg-0pky">-8.7843</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">InstagScorer</td>
      <td class="tg-0pky">InstagScore</td>
      <td class="tg-0pky">Evaluates the content diversity by returning the number of tags. <b>The higher the score, the greater the content diversity.</b></td>
      <td class="tg-0pky">2.304</td>
      <td class="tg-0pky">2.9396</td>
      <td class="tg-0pky">11</td>
      <td class="tg-0pky">1</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">SuperfilteringScorer</td>
      <td class="tg-0pky">SuperfilteringScore</td>
      <td class="tg-0pky">Evaluates the instruction-following difficulty using the Superfiltering method. <b>The higher the score, the more difficult it is to follow the instructions.</b></td>
      <td class="tg-0pky">1.3223</td>
      <td class="tg-0pky">836.0302</td>
      <td class="tg-0pky">1978.6534</td>
      <td class="tg-0pky">0.0011</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">DeitaQualityScorer</td>
      <td class="tg-0pky">DeitaQualityScore</td>
      <td class="tg-0pky">Instruction quality evaluation based on the Llama model. <b>The higher the score, the better the quality of the instructions.</b></td>
      <td class="tg-0pky">3.5629</td>
      <td class="tg-0pky">0.9247</td>
      <td class="tg-0pky">5.5309</td>
      <td class="tg-0pky">1.0840</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">DeitaComplexityScorer</td>
      <td class="tg-0pky">DeitaComplexityScore</td>
      <td class="tg-0pky">Instruction complexity evaluation based on the Llama model. <b>The higher the score, the greater the complexity of the instructions.</b></td>
      <td class="tg-0pky">1.4936</td>
      <td class="tg-0pky">0.2086</td>
      <td class="tg-0pky">3.3207</td>
      <td class="tg-0pky">1.0001</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="3">VendiScorer</td>
      <td class="tg-0pky">N-grams_VendiScore</td>
      <td class="tg-0pky">Evaluates text diversity based on N-grams embeddings. <b>The higher the score, the greater the dataset diversity.</b></td>
      <td class="tg-0pky">1832.96</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
  </tr>
  <tr>
      <td class="tg-0pky">BERT_VendiScore</td>
      <td class="tg-0pky">Evaluates text diversity based on BERT embeddings. <b>The higher the score, the greater the dataset diversity.</b></td>
      <td class="tg-0pky">1.83</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
  </tr>
  <tr>
      <td class="tg-0pky">SimCSE_VendiScore</td>
      <td class="tg-0pky">Evaluates text diversity based on SimCSE embeddings. <b>The higher the score, the greater the dataset diversity.</b></td>
      <td class="tg-0pky">68.94</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">Task2VecScorer</td>
      <td class="tg-0pky">Task2VecScore</td>
      <td class="tg-0pky">Evaluates dataset diversity using Task2Vec diversity coefficient. <b>The higher the score, the greater the dataset diversity.</b></td>
      <td class="tg-0pky">0.0673</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
  </tr>
  <tr>
      <td class="tg-0pky" rowspan="1">AlpagasusScorer</td>
      <td class="tg-0pky">AlpagasusScore</td>
      <td class="tg-0pky">Evaluates instruction quality using ChatGPT. <b>The higher the score, the better the quality of the instructions.</b></td>
      <td class="tg-0pky">4.172</td>
      <td class="tg-0pky">0.2164</td>
      <td class="tg-0pky">5.0</td>
      <td class="tg-0pky">2.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">TreeinstructScorer</td>
    <td class="tg-0pky">TreeinstructScore</td>
    <td class="tg-0pky">Uses ChatGPT to evaluate the semantic complexity of instructions. <b>The higher the score, the greater the semantic complexity of the instruction.</b></td>
    <td class="tg-0pky">6.494</td>
    <td class="tg-0pky">9.7540</td>
    <td class="tg-0pky">63.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">PresidioScorer</td>
    <td class="tg-0pky">PresidioScore</td>
    <td class="tg-0pky">Uses Presidio to evaluate the number of PII (Personally Identifiable Information) instances. <b>The higher the score, the more PII information is present in the text.</b></td>
    <td class="tg-0pky">21.4008</td>
    <td class="tg-0pky">2915.3542</td>
    <td class="tg-0pky">1786.0</td>
    <td class="tg-0pky">0.0</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="1">RMScorer</td>
    <td class="tg-0pky">RMScore</td>
    <td class="tg-0pky">Uses a reward model based on human values to evaluate the quality of SFT (Supervised Fine-Tuning) data. <b>The higher the score, the better the data quality.</b></td>
    <td class="tg-0pky">3.1537</td>
    <td class="tg-0pky">9.9461</td>
    <td class="tg-0pky">8.6803</td>
    <td class="tg-0pky">-4.9680</td>
  </tr>
</tbody>
</table>
