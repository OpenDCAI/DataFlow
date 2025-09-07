from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.special import softmax
from dataflow import get_logger
import torch
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class DeitaComplexitySampleEvaluator(OperatorABC):
    def __init__(self, device='cuda', model_cache_dir='./dataflow_cache', max_length=512):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "hkust-nlp/deita-complexity-scorer"
        self.model_cache_dir = model_cache_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.score_name = 'DeitaComplexityScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于Llama模型的Deita指令复杂性评估器，通过生成1-6分的复杂性评分评估指令难度。\n"
                "输入参数：\n"
                "- device：计算设备，默认为'cuda'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- max_length：最大序列长度，默认为512\n"
                "- input_instruction_key：指令文本字段名，默认为'instruction'\n"
                "- input_output_key：输出文本字段名，默认为'output'\n"
                "- output_key：输出得分字段名，默认为'DeitaComplexityScore'\n"
                "输出参数：\n"
                "- 包含指令复杂性评分的DataFrame（1-6分）"
            )
        elif lang == "en":
            return (
                "Llama-based Deita instruction complexity evaluator generating 1-6 complexity scores.\n"
                "Input Parameters:\n"
                "- device: Computing device, default 'cuda'\n"
                "- model_cache_dir: Model cache directory, default './dataflow_cache'\n"
                "- max_length: Maximum sequence length, default 512\n"
                "- input_instruction_key: Field name for instruction text, default 'instruction'\n"
                "- input_output_key: Field name for output text, default 'output'\n"
                "- output_key: Field name for output score, default 'DeitaComplexityScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing instruction complexity scores (1-6)"
            )
        else:
            return "Measure instruction complexity using Llama-based Deita model."

    def infer_complexity(self, input_text):
        complexity_template = ("You are a helpful assistant. Please identify the complexity score of the following user query. \n##Query: {instruction}\n##Complexity: ")
        user_input = complexity_template.format(instruction=input_text)
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids, max_new_tokens=self.max_length, num_return_sequences=1, return_dict_in_generate=True, output_scores=True)
        logprobs_list = outputs.scores[0][0]
        # Mapping of token IDs to complexity scores
        id2score = {
            29896: 1,  # Complexity level 1
            29906: 2,  # Complexity level 2
            29941: 3,  # Complexity level 3
            29946: 4,  # Complexity level 4
            29945: 5,  # Complexity level 5
            29953: 6   # Complexity level 6
        }
        score_template = np.array([1, 2, 3, 4, 5, 6])  # Define the score template
        score_logits = []
        for k in id2score:
            score_logits.append(logprobs_list[k].cpu().numpy())

        score_logits = np.array(score_logits)
        score_npy = softmax(score_logits, axis=0)  # Apply softmax to get probabilities
        score_npy = score_npy * score_template  # Weight the scores by the corresponding complexity level
        final_score = np.sum(score_npy, axis=0)  # Sum the weighted scores to get the final score
        return final_score

    def eval(self, dataframe, input_instruction_key: str = 'instruction', input_output_key: str = 'output'):
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = []
        for sample in tqdm(dataframe[[input_instruction_key, input_output_key]].to_dict(orient='records'), desc="Deita complexity model evaluating..."):
            quality_score = self.infer_complexity(sample[input_instruction_key])
            scores.append(quality_score)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'DeitaComplexityScore'):

        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key, input_output_key)
        dataframe[output_key] = scores        
        storage.write(dataframe)
