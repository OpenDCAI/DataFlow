from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataflow import get_logger
import json

@OPERATOR_REGISTRY.register()
class InstagSampleEvaluator(OperatorABC):
    def __init__(self, model_cache_dir='./dataflow_cache', device='cuda', max_new_tokens=1024, temperature=0, do_sample=False, num_return_sequences=1, return_dict_in_generate=True):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'OFA-Sys/InsTagger'
        self.model_cache_dir = model_cache_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.return_dict_in_generate = return_dict_in_generate
        self.token_strs = ["1", "2", "3", "4", "5", "6"]
        self.score_template = np.array([1, 2, 3, 4, 5, 6])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.score_name = 'InstagScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用Instag评分器评估指令的内容多样性和意图标签。通过分析指令文本生成相关标签，标签数量越多表示内容多样性越大，" 
                "同时返回标签的详细解释。基于OFA-Sys/InsTagger模型实现。\n" 
                "输入参数：\n" 
                "- query: 待评估的指令文本\n" 
                "输出参数：\n" 
                "- int: 标签数量（内容多样性指标）\n" 
                "- list: 包含标签和解释的字典列表"
            )
        else:
            return (
                "Evaluate instruction content diversity and intention tags using the Instag scorer. Generate relevant tags by analyzing instruction text, " 
                "with more tags indicating greater content diversity, while returning detailed explanations of tags. Implemented based on OFA-Sys/InsTagger model.\n" 
                "Input parameters:\n" 
                "- query: Instruction text to be evaluated\n" 
                "Output parameters:\n" 
                "- int: Number of tags (content diversity indicator)\n" 
                "- list: List of dictionaries containing tags and explanations"
            )

    def make_prompt(self, query):
        prompt = f"Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}.\nUser query: {query}"
        messages = [("user", prompt), ("Assistant", None)]
        seps = [" ", "</s>"]
        ret = "system: A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def inference_batch(self, queries):
        """Process batch of queries using either local model or API."""
        input_strs = [self.make_prompt(query) for query in queries]
        input_tokens = self.tokenizer(input_strs, return_tensors="pt", padding=True)

        if torch.cuda.is_available():
            input_tokens = {key: value.to(self.device) for key, value in input_tokens.items()}

        output = self.model.generate(
            input_tokens['input_ids'],
            temperature=self.temperature,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_return_sequences,
            return_dict_in_generate=self.return_dict_in_generate,
        )
        
        num_input_tokens = input_tokens["input_ids"].shape[1]
        output_tokens = output.sequences
        generated_tokens = output_tokens[:, num_input_tokens:]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        json_outputs = []
        for generated_text in generated_texts:
            string_output = generated_text.strip()
            try:
                json_output = json.loads(string_output)
            except json.JSONDecodeError:
                self.logger.warning(f"JSON parse error: {string_output}")
                json_output = {"tag": "Parsing error", "explanation": string_output[:100]}
            json_outputs.append(json_output)
        
        return json_outputs

    def _score_func(self, sample, input_instruction_key):
        json_output = self.inference_batch([sample])[0]
        complexity_score = None
        if isinstance(json_output, list):
            complexity_score = len(json_output)
            self.logger.info(f"列表类型JSON,标签数量: {complexity_score}")
        elif isinstance(json_output, dict) and "tag" in json_output: 
            complexity_score = 1
            self.logger.info(f"字典类型JSON,包含tag字段,评分为1")
        elif isinstance(json_output, dict) and len(json_output) > 0: 
            complexity_score = 1
            self.logger.info(f"其他字典类型JSON,评分为1: {json_output}")
        else:
            complexity_score = 0
            self.logger.warning(f"未识别的JSON类型或空数据,评分为0: {json_output}") 
        return complexity_score

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str):
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = []
        for sample in tqdm(dataframe[input_instruction_key], desc="Instagger mode evaluating..."):
            scores.append(self._score_func(sample, input_instruction_key))
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', output_key: str = 'InstagScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key)
        dataframe[output_key] = scores
        storage.write(dataframe)
