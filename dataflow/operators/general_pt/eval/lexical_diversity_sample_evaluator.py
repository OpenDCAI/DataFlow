from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import string
from tqdm import tqdm

remove_punctuation = str.maketrans('', '', string.punctuation)

# mtld, hdd and other functions remain unchanged as they are utility functions
# ... (mtld_calc, mtld, factorial, combination, hypergeometric, hdd)

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


@OPERATOR_REGISTRY.register()
class LexicalDiversitySampleEvaluator(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.metrics_to_keep = {'mtld': True, 'hdd': True}
        self.score_name = 'LexicalDiversityScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用MTLD（词汇多样性测量）和HDD（移动平均类型-标记比）方法计算文本词汇多样性。\n\n"
                "功能说明：\n"
                "- MTLD（词汇多样性测量）：通过计算维持特定TTR阈值所需的单词数量来评估词汇多样性\n"
                "- HDD（移动平均类型-标记比）：基于样本的词汇丰富度估计\n\n"
                "输入要求：文本长度需大于50个单词\n\n"
                "输出参数：\n"
                "- LexicalDiversityMTLDScore: MTLD多样性得分（值越高表示多样性越好）\n"
                "- LexicalDiversityHD-DScore: HDD多样性得分（值越高表示多样性越好）"
            )
        else:
            return (
                "Measures text lexical diversity using MTLD (Measure of Textual Lexical Diversity) and HDD (Hypergeometric Distribution Diversity) methods.\n\n"
                "Features:\n"
                "- MTLD: Evaluates lexical diversity by calculating the number of words needed to maintain a specific TTR threshold\n"
                "- HDD: Estimates vocabulary richness based on sampling\n\n"
                "Input Requirement: Text length must exceed 50 words\n\n"
                "Output Parameters:\n"
                "- LexicalDiversityMTLDScore: MTLD diversity score (higher = more diverse)\n"
                "- LexicalDiversityHD-DScore: HDD diversity score (higher = more diverse)"
            )

    def _score_func(self, sample):
        text = sample
        words = text.split()
        scores = {}
        # must ensure text length in the given interval
        if self.metrics_to_keep.get('mtld'):
            if len(words) > 50:
                scores['LexicalDiversityMTLDScore'] = mtld(words)
            else:
                scores['LexicalDiversityMTLDScore'] = None

        if self.metrics_to_keep.get('hdd'):
            if 50 < len(words) < 1000:
                scores['LexicalDiversityHD-DScore'] = hdd(words)
            else:
                scores['LexicalDiversityHD-DScore'] = None

        return scores

    def eval(self, dataframe, input_key):
        scores_list = []
        self.logger.info(f"Evaluating {self.score_name}...")
        for sample in tqdm(dataframe[input_key], desc="LexicalDiversityScore Evaluating..."):
            scores = self._score_func(sample)
            scores_list.append(scores)
        self.logger.info("Evaluation complete!")
        return scores_list
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info("LexicalDiversityScore ready to evaluate.")
        
        scores = self.eval(dataframe, input_key)
        # Flatten the nested dictionary of scores into the dataframe
        for idx, score_dict in enumerate(scores):
            for key, value in score_dict.items():
                dataframe.at[idx, key] = value
        storage.write(dataframe)

