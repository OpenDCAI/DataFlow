from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_sft.eval.Superfiltering.data_analysis import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.utils import get_logger
import pandas as pd

# Superfiltering instruction quality (ifd) evaluation
# cited from: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
@OPERATOR_REGISTRY.register()
class SuperfilteringSampleEvaluator(OperatorABC):
    def __init__(self, device='cuda', model_cache_dir='./dataflow_cache', max_length=512):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.device = device
        self.model_name = 'gpt2'
        self.model_cache_dir = model_cache_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=self.device, 
            cache_dir=self.model_cache_dir, 
            output_hidden_states=True
        ).to(self.device)
        self.score_name = 'SuperfilteringScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用Superfiltering方法评估指令的跟随难度，基于GPT-2模型计算条件困惑度与独立困惑度的比值，得分越高表示指令越难跟随。" 
                "该方法通过比较指令条件下的响应困惑度与独立响应困惑度，评估指令的清晰度和跟随难度。\n" 
                "输入参数：\n" 
                "- instruction: 指令文本\n" 
                "- input_text: 输入文本（可选）\n" 
                "- output: 响应文本\n"
                "- input_instruction_key：run()读取的指令字段名，默认为'instruction'\n"
                "- input_input_key：run()读取的输入文本字段名，默认为None（不使用输入文本）\n"
                "- input_output_key：run()读取的响应字段名，默认为'output'\n"
                "- output_key：写入得分的输出字段名，默认为'SuperfilteringScore'\n"
                "输出参数：\n"
                "- float: 困惑度比值，越高表示指令跟随难度越大；响应为空或结果为NaN时为None，独立困惑度为0时为0\n"
                "- 不过滤任何行；每行新增output_key字段，值为该行的浮点得分（可能为None），该字段保留在输出中；run()无返回值(None)"
            )
        else:
            return (
                "Evaluate the follow difficulty of instructions using the Superfiltering method, which calculates the ratio of conditional perplexity to independent perplexity based on the GPT-2 model. " 
                "Higher scores indicate greater difficulty in following the instruction. This method assesses instruction clarity and follow difficulty by comparing response perplexity under instruction conditions with independent response perplexity.\n" 
                "Input parameters:\n" 
                "- instruction: Instruction text\n" 
                "- input_text: Input text (optional)\n" 
                "- output: Response text\n"
                "- input_instruction_key: Name of the instruction field read by run(), default is 'instruction'\n"
                "- input_input_key: Name of the input text field read by run(), default is None (no input text used)\n"
                "- input_output_key: Name of the response field read by run(), default is 'output'\n"
                "- output_key: Name of the output field that receives the score, default is 'SuperfilteringScore'\n"
                "Output parameters:\n"
                "- float: Perplexity ratio, higher values indicate greater instruction following difficulty; None when the response is empty or the result is NaN, 0 when the standalone perplexity is 0\n"
                "- No rows are filtered; each row gets a new output_key field holding its float score (may be None), which is kept in the output; run() returns None"
            )

    def inference(self, instruction, input_text, output):
        PROMPT_DICT_NONE = {
            "prompt_input": (
                "{instruction}\n{input}\n"
            ),
            "prompt_no_input": (
                "{instruction}\n"
            ),
        }
        prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
        prompt_input = PROMPT_DICT_NONE["prompt_input"]

        if input_text == '':
            temp_dict = {'instruction': instruction}
            prompt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use
        else:
            temp_dict = {'instruction': instruction, 'input': input_text}
            prompt_to_use = prompt_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use

        if output == '':
            return None
            
        instruction_input_ids = self.tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        instruction_len = instruction_input_ids.shape[1]
        
        ppl_out_alone, _ = get_perplexity_and_embedding_whole_text(self.tokenizer, self.model, output, self.max_length - instruction_len + 1, self.device)
        ppl_out_condition, _ = get_perplexity_and_embedding_part_text(self.tokenizer, self.model, whole_text, output, self.max_length, self.device)

        if ppl_out_alone != 0:
            score = ppl_out_condition / ppl_out_alone
        else:
            score = 0

        if score != score:  # 检查NaN
            score = None
            
        return score

    def _score_func(self, sample, input_instruction_key: str = 'instruction', input_input_key: str = 'input', input_output_key: str = 'output'):
        instruction = sample.get(input_instruction_key, [''])
        output = sample.get(input_output_key, [''])
        input_text = sample.get(input_input_key, ['']) if input_input_key is not None and input_input_key in sample else ''
        
        if not output:
            score = None
        else:
            score = self.inference(instruction, input_text, output)
            
        return score
    
    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str = 'instruction', input_input_key: str = None, input_output_key: str = 'output'):
        self.logger.info(f"Evaluating {self.score_name}...")
        key_list = [input_instruction_key, input_output_key]
        if input_input_key is not None:
            key_list.append(input_input_key)
        scores = [self._score_func(sample, input_instruction_key, input_input_key, input_output_key) for sample in tqdm(dataframe[key_list].to_dict(orient='records'), desc="SuperfilteringScorer evaluating...")]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_input_key: str = None, input_output_key: str = 'output', output_key: str = 'SuperfilteringScore'):
        dataframe = storage.read("dataframe") 
        scores = self.eval(dataframe, input_instruction_key, input_input_key, input_output_key)
        dataframe[output_key] = scores
        storage.write(dataframe)
