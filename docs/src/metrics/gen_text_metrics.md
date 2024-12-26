# Evaluation Metrics for Generated Text

## Overview

| Category        | Number of Scorers | Description                                         |
| --------------- | ----------------- | --------------------------------------------------- |
| Word Overlap Based | 5               | Evaluates the n-gram overlap between generated and reference texts |
| Word Embeddings Based | 2           | Uses word embeddings to calculate similarity between generated and reference texts |
| Language Models Based | 4          | Utilizes pre-trained language models to evaluate semantics and fluency |
| Others          | 2                 | -                                                   |

---

## Word Overlap Based

| Scorer Name          | Evaluation Dimension | Application Scenario | Implementation                                                                                 | Value Range     | Interpretation                                | Advantages                        | Limitations                      |
| -------------------- | -------------------- | -------------------- | --------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------- | --------------------------------- | -------------------------------- |
| BLEU Scorer          | Fluency and Lexical Match | Machine Translation, Text Generation | Calculates precision based on n-gram matching by comparing n-grams in generated and reference texts | [0, 1]         | Higher values indicate greater match between generated and reference texts | Suitable for large datasets, simple and efficient | Performs poorly at sentence level, insensitive to synonyms and word order |
| ROUGE Scorer         | Content Overlap      | Text Summarization   | Calculates overlap between generated and reference summaries using n-gram and longest common subsequence matching | [0, 1]         | Higher values indicate more content overlap between generated and reference texts | Easy to use, applicable to various text generation tasks | Limited semantic understanding |
| METEOR Scorer        | Semantic Matching    | Machine Translation  | Calculates alignment scores between generated and reference texts based on stemming, synonym matching, and semantic relevance | [0, 1]         | Higher values indicate stronger semantic consistency between generated and reference texts | More sensitive to semantic similarity than BLEU, closer to human evaluation | High computational complexity |
| CIDEr Scorer         | Content Relevance   | Image Caption Generation | Uses TF-IDF weighted n-gram statistics to compare similarity between generated and reference descriptions | [0, 1]         | Higher values indicate stronger content consistency between generated and reference texts | Considers the weight of words in the reference text, suitable for image-to-text tasks | Strong influence of low-frequency words |
| CHRF Scorer | Lexical Matching | Machine Translation | Calculates the chrF score based on character-level n-gram precision and recall between the reference text and the evaluated text | [0, 1] | The higher the value, the stronger the semantic similarity | Allows for more fine-grained understanding | Ignores semantic information |
---

### Word Embeddings Based

| Scorer Name                  | Evaluation Dimension | Application Scenario | Implementation                                                                                 | Value Range     | Interpretation                                | Advantages                        | Limitations                      |
| --------------------------- | -------------------- | -------------------- | --------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------- | --------------------------------- | -------------------------------- |
| Embedding Average Score      | Semantic Similarity  | Text Generation      | Computes cosine similarity of the average word embeddings of generated and reference texts     | [0, 1]         | Higher values indicate stronger semantic similarity | Simple and efficient, suitable for quick computations | Cannot capture complex semantic structures |
| Greedy Matching Score        | Semantic Relevance   | Text Generation      | Matches semantically similar words between generated and reference texts, computes similarity   | [0, 1]         | Higher values indicate stronger semantic relevance | Captures local similarity                     | Ignores global semantic structure            |
---

### Language Models Based

| Scorer Name          | Evaluation Dimension | Application Scenario | Implementation                                                                                 | Value Range     | Interpretation                                | Advantages                        | Limitations                      |
| ---------------------| -------------------- | -------------------- | --------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------- | --------------------------------- | -------------------------------- |
| WSD Scorer                   | Semantic Similarity  | Text Generation, Semantic Similarity Analysis | Uses word2vec to calculate Word Mover's Distance (WMD) between generated and reference texts   | [0, +∞)        | Lower values indicate closer semantic distance between generated and reference texts | Captures deep semantic differences, applicable to various languages | Sensitive to text length and stopwords, high computational complexity |
| BertScore            | Semantic Similarity  | Text Generation      | Computes similarity of word embeddings between generated and reference texts using BERT        | [0, 1]         | Higher values indicate stronger semantic similarity between generated and reference texts | Captures deep semantic information, supports multiple languages | Depends on pre-trained models, time-consuming computations |
| BARTScore            | Fluency and Informativeness | Text Generation      | Uses BART model to treat the generated text as the target and computes the likelihood score     | (-∞, +∞)       | Higher values indicate better quality of generated text | Provides a multi-dimensional evaluation of text quality | Strong dependency on models |
| BELURT Scorer        | Semantic Similarity  | Text Generation, Machine Translation | Fine-tunes pre-trained language models (e.g., BERT) for semantic similarity tasks, computes similarity scores between generated and reference texts | [0, 1]         | Higher values indicate stronger semantic consistency between generated and reference texts | Combines semantic understanding of pre-trained models, captures deep semantic information | Model training depends on high-quality data, sensitive to domain changes, and computationally expensive |

---

### Others

| Scorer Name          | Evaluation Dimension | Application Scenario | Implementation                                                                                 | Value Range     | Interpretation                                | Advantages                        | Limitations                      |
| ---------------------| -------------------- | -------------------- | --------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------- | --------------------------------- | -------------------------------- |
| TER Scorer           | Edit Distance       | Machine Translation  | Calculates the minimum edit operations (insertions, deletions, and substitutions) needed to transform generated text into reference text | [0, 1]         | Lower values indicate closer match between generated and reference texts | Simple and intuitive, suitable for analyzing errors in machine translation | Insensitive to semantic information |
| HLEPOR Scorer        | Multi-Dimensional Matching | Machine Translation  | Calculates multi-dimensional matching scores between generated and reference texts considering multiple weighted parameters (e.g., position, proportion) | [0, 1]         | Higher values indicate stronger match between generated and reference texts | Highly flexible, adjustable weight parameters to fit different tasks | Parameter selection significantly affects evaluation results |
