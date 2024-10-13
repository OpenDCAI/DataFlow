from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import string

# Text Lexical diversity evaluation
# cited from: MTLD, vocd-D, and HDD: A validation study of sophisticated approaches to lexical diversity assessment


#Copyright 2017 John Frens
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
remove_punctuation = str.maketrans('', '', string.punctuation)

def mtld_calc(word_array, ttr_threshold):
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0
    
    for token in word_array:
        token = token.translate(remove_punctuation).lower() 
        token_count += 1
        if token not in types:
            type_count +=1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0
    
    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1

def mtld(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("The input should be a list of str")
    if len(word_array) < 50:
        raise ValueError("The input length should be larger than 50")
    return (mtld_calc(word_array, ttr_threshold) + mtld_calc(word_array[::-1], ttr_threshold)) / 2


def factorial(x):
    x=int(x)
    result = 1
    for i in range(2, x + 1):
        result *= i
    return result

def combination(n, r):
    r_fact = factorial(r)
    numerator = 1.0
    num = n-r+1.0
    while num < n+1.0:
        numerator *= num
        num += 1.0
    return numerator / r_fact

def hypergeometric(population, population_successes, sample, sample_successes):
    return (combination(population_successes, sample_successes) *
            combination(population - population_successes, sample - sample_successes)) /\
            combination(population, sample)

def hdd(word_array, sample_size=42.0):
    if isinstance(word_array, str):
        raise ValueError("The input should be a list of str")
    if len(word_array) < 50:
        raise ValueError("The input length should be larger than 50")

    type_counts = {}
    for token in word_array:
        token = token.translate(remove_punctuation).lower()  
        if token in type_counts:
            type_counts[token] += 1.0
        else:
            type_counts[token] = 1.0

    hdd_value = 0.0
    for token_type in type_counts.keys():
        contribution = (1.0 - hypergeometric(len(word_array), sample_size, type_counts[token_type], 0.0)) / sample_size
        hdd_value += contribution

    return hdd_value

@MODEL_REGISTRY.register()
class LexicalDiversityScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.metrics_to_keep = args_dict.get('metrics_to_keep', {})
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'LexicalDiversityScore' 

    def evaluate_batch(self, batch):
        results = {}
        input_data = next(iter(batch.values()))

        for sample in input_data:
            text = sample 
            words = text.split()

            if self.metrics_to_keep.get('mtld'):
                mtld_score = mtld(words) if len(words) > 50 else None
                results.setdefault('LexicalDiversityMTLDScore', []).append(mtld_score)

            if self.metrics_to_keep.get('hdd'):
                hdd_score = hdd(words) if 50 < len(words) < 1000 else None
                results.setdefault('LexicalDiversityHD-DScore', []).append(hdd_score)

        return results 
