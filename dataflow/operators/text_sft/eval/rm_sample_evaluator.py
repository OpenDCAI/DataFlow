from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
import torch
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.utils import get_logger

# RMScorer for evaluating based on reward-model-deberta-v3-large-v2
@OPERATOR_REGISTRY.register()
class RMSampleEvaluator(OperatorABC):
    def __init__(self, device='cuda', model_cache_dir='./dataflow_cache', ):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
        self.model_cache_dir = model_cache_dir
        self.score_name = 'RewardModelScore'
        self.device = device
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于人类偏好数据训练的奖励模型(OpenAssistant/reward-model-deberta-v3-large-v2)对文本质量进行打分，高分代表质量较高。" 
                "模型输入为指令和响应文本对，输出奖励模型的原始logit作为奖励分数（无sigmoid或截断，不保证落在0-1之间），反映人类对文本质量的偏好判断。\n"
                "输入参数：\n"
                "- instruction: 指令文本字符串\n"
                "- output: 响应文本字符串\n"
                "- input_instruction_key：run()读取的指令字段名，默认为'instruction'\n"
                "- input_output_key：run()读取的响应字段名，默认为'output'\n"
                "- output_key：写入奖励分数的输出字段名，默认为'RMScore'\n"
                "输出参数：\n"
                "- float: 奖励分数（原始logit），越高表示质量越好\n"
                "- 不过滤任何行；每行新增output_key字段，值为该行的浮点奖励分数，该字段保留在输出中；run()无返回值(None)"
            )
        else:
            return (
                "Score text quality using a reward model trained on human preference data (OpenAssistant/reward-model-deberta-v3-large-v2), where higher scores indicate better quality. " 
                "The model takes instruction-response text pairs as input and outputs the reward model's raw logit as the reward score (no sigmoid or clipping, so it is not guaranteed to lie in 0-1), reflecting human preference judgments on text quality.\n"
                "Input parameters:\n"
                "- instruction: Instruction text string\n"
                "- output: Response text string\n"
                "- input_instruction_key: Name of the instruction field read by run(), default is 'instruction'\n"
                "- input_output_key: Name of the response field read by run(), default is 'output'\n"
                "- output_key: Name of the output field that receives the reward score, default is 'RMScore'\n"
                "Output parameters:\n"
                "- float: Reward score (raw logit), higher values indicate better quality\n"
                "- No rows are filtered; each row gets a new output_key field holding its float reward score, which is kept in the output; run() returns None"
            )

    def eval(self, dataframe, input_instruction_key: str = 'instruction', input_output_key: str = 'output'):
        input_texts = dataframe.get(input_instruction_key, '').to_list()
        output_texts = dataframe.get(input_output_key, '').to_list()
        inputs = self.tokenizer(input_texts, output_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        self.logger.info(f"Evaluating {self.score_name}...")
        with torch.no_grad():
            logits = self.rank_model(**inputs).logits.cpu().detach().numpy()
        scores = logits.squeeze() 
        if scores.ndim == 0:  
            scores = [float(scores)]
        self.logger.info("Evaluation complete!")
        return scores.tolist() 

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'RMScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key, input_output_key)
        dataframe[output_key] = scores        
        storage.write(dataframe)