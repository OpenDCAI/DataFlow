# Text Data Processing

## Overview

DataFlow currently supports text data processing at the data point level, categorized into three types: refiners, deduplicators, and filters.

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
      <td class="tg-0pky">Refiners</td>
      <td class="tg-0pky">15</td>
      <td class="tg-0pky">Improves the content of data points through processing and augmentation without altering the total count.</td>
    </tr>
    <tr>
      <td class="tg-0pky">Deduplicators</td>
      <td class="tg-0pky">6</td>
      <td class="tg-0pky">Removes duplicate data points using methods such as hashing.</td>
    </tr>
    <tr>
      <td class="tg-0pky">Filters</td>
      <td class="tg-0pky">42</td>
      <td class="tg-0pky">Filters data points based on thresholds and other criteria.</td>
    </tr>
  </tbody>
</table>

## Refiners

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Applicable Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LowercaseRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Converts text fields to lowercase.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">PIIAnonymizeRefiner</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Anonymizes Personally Identifiable Information (PII), such as names and locations, to protect privacy.</td>
      <td class="tg-0pky"><a href="https://github.com/microsoft/presidio">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemovePunctuationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Removes punctuation from text.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveNumberRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Removes numeric characters from text.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveExtraSpacesRefiner</td>
      <td class="tg-0pky">NLP, Pre-training</td>
      <td class="tg-0pky">Replaces multiple consecutive spaces with a single space and trims leading/trailing spaces.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveRepetitionsPunctuationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Removes repeated punctuation, e.g., "!!!" becomes "!".</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveEmojiRefiner</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Removes emojis from text, e.g., "😀".</td>
      <td class="tg-0pky"><a href="https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveEmoticonsRefiner</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Removes emoticons such as ":-)", using a predefined list.</td>
      <td class="tg-0pky"><a href="https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveContractionsRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Expands contractions in text, e.g., "can't" becomes "cannot".</td>
      <td class="tg-0pky"><a href="https://github.com/kootenpv/contractions">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">HtmlUrlRemoverRefiner</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Removes URLs and HTML tags from text.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">TextNormalizationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Normalizes formats for dates, currencies, etc., in text.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">NERRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Uses Named Entity Recognition (NER) to identify and mask specific entities in text.</td>
      <td class="tg-0pky"><a href="https://spacy.io/usage/linguistic-features#named-entities">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">StemmingLemmatizationRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Performs stemming or lemmatization on text.</td>
      <td class="tg-0pky"><a href="https://www.nltk.org/">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SpellingCorrectionRefiner</td>
      <td class="tg-0pky">NLP, Pre-training</td>
      <td class="tg-0pky">Corrects spelling errors in text using SymSpell.</td>
      <td class="tg-0pky"><a href="https://github.com/mammothb/symspellpy">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">RemoveStopwordsRefiner</td>
      <td class="tg-0pky">NLP</td>
      <td class="tg-0pky">Removes stopwords (e.g., "the", "is") from text.</td>
      <td class="tg-0pky"><a href="https://github.com/nltk/nltk">Code</a></td>
    </tr>
  </tbody>
</table>

## Deduplicators

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Applicable Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">HashDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">Uses various hash functions (e.g., MD5, SHA256, XXH3_128) to remove duplicate data based on exact hash value comparison. Suitable for small-scale simple deduplication.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">CCNetDeduplicator</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Compares the first 64 bits of the SHA-1 hash to identify duplicate text, balancing security and computational efficiency.</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">NgramHashDeduplicator</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Combines n-gram techniques with hashing to detect duplicates based on multiple hash comparisons of n-gram segments. Useful for identifying near-duplicates.</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1607.04606">Paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SemDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">Uses semantic similarity based on BERT embeddings and cosine similarity to detect duplicates. Ideal for detecting semantically similar but differently phrased text.</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1810.04805">Paper</a> <br> <a href="https://github.com/facebookresearch/SemDeDup">Code</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">SimHashDeduplicator</td>
      <td class="tg-0pky">Pre-training</td>
      <td class="tg-0pky">Uses the SimHash algorithm to detect similar text based on Hamming distance of fingerprints. Efficient for large-scale data deduplication.</td>
      <td class="tg-0pky"><a href="https://dl.acm.org/doi/abs/10.1145/1242572.1242592">Paper</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">MinHashDeduplicator</td>
      <td class="tg-0pky">SFT</td>
      <td class="tg-0pky">Combines MinHash and LSH to compare sets with minimal memory usage and computation cost, detecting similarity between sets.</td>
      <td class="tg-0pky"><a href="https://arxiv.org/abs/1811.04633">Paper</a></td>
    </tr>
  </tbody>
</table>

## Filters

<table class="tg">
  <thead>
    <tr>
      <th class="tg-0pky">Name</th>
      <th class="tg-0pky">Applicable Type</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Repository or Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">LanguageFilter</td>
      <td class="tg-0pky">Pre-training, SFT</td>
      <td class="tg-0pky">Filters specific languages using the fasttext language identification model.</td>
      <td class="tg-0pky"><a href="https://huggingface.co/facebook/fasttext-language-identification">Huggingface</a></td>
    </tr>
    <tr>
      <td class="tg-0pky">BlocklistFilter</td>
      <td class="tg-0pky">Pre-training, SFT</td>
      <td class="tg-0pky">Filters data points using a blocklist (e.g., List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words).</td>
      <td class="tg-0pky"><a href="https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words?tab=readme-ov-file">Code</a></td>
    </tr>
  </tbody>
</table>

Additionally, Open-DataFlow-Eval supports filtering data points based on scores from single data point scorers, with 18 supported scorers.

```yaml
DeitaQualityFilter:
    min_score: 1                                         
    max_score: 5                                      
    scorer_args:
      device: 'cuda:0'
      model_name: 'hkust-nlp/deita-quality-scorer'
      max_length: 512
```
You can set min/max scores and scorer parameters in `scorer_args` for filtering. For more information on supported scorers, refer to the [evaluation algorithm documentation](text_metrics.md) (excluding the Diversity part).

In addition, heuristic rule filtering plays a significant role in the screening of pre-training data. In this regard, the [Dingo Data Quality Evaluation Tool](https://github.com/DataEval/dingo) has greatly inspired our development. We have integrated some of the rule filtering algorithms used in Dingo, a total of 22 types, into `dataflow/process/text/filters/heuristics.py`. For details, please refer to the [Rules Documentation](https://github.com/DataEval/dingo/blob/dev/docs/rules.md). The names of the filters can be found in the `dataflow/process/text/filters/heuristics.py` file.


All 42 data filters mentioned above share the same `yaml` invocation method.
