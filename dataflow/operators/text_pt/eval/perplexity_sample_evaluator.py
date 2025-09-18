import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.utils import get_logger

@OPERATOR_REGISTRY.register()
class PerplexitySampleEvaluator(OperatorABC):
    def __init__(self, model_name: str = 'gpt2', device='cuda'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = model_name
        self.score_name = 'PerplexityScore'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load Hugging Face model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            self.logger.info(f'{self.__class__.__name__} initialized with model {self.model_name}.')
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Model loading failed. Please ensure the model is available from Hugging Face.")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于Huggingface语言模型计算文本的困惑度(Perplexity)，困惑度越低表示文本的流畅性和可理解性越高。" 
                "输入参数：\n" 
                "- model_name：Huggingface模型路径或名称\n"
                "- device：模型运行设备\n"
                "输出参数：\n" 
                "- float: 困惑度值，越低表示文本流畅性越好"
            )
        else:
            return (
                "Calculate text perplexity using a Huggingface language model; lower perplexity indicates better fluency and understandability."
                "Input Parameters:\n"
                "- model_name: Huggingface model path or name\n"
                "- device: Model device\n\n"
                "Output Parameters:\n"
                "- float: Perplexity score, lower values indicate better fluency and understandability"
            )

    def eval(self, dataframe, input_key):
        input_texts = dataframe.get(input_key, '').to_list()
        self.logger.info(f"Evaluating {self.score_name}...")
        results = []
        
        # Use tqdm to show progress
        for text in tqdm(input_texts, desc="Evaluating perplexity", unit="text"):
            perplexity = self.calculate_perplexity(text)
            results.append(perplexity)
        
        self.logger.info("Evaluation complete!")
        return results

    def calculate_perplexity(self, text: str) -> float:
        """ 使用Hugging Face模型计算困惑度 """
        # Encode the input text
        inputs = self.tokenizer(text, return_tensors='pt', padding="longest", truncation=True).to(self.device)
        # Calculate log probability
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            log_likelihood = outputs.loss * inputs['input_ids'].size(1)

        # Perplexity calculation formula: exp(log_prob / N) -> Perplexity = exp(-average log probability)
        perplexity = torch.exp(log_likelihood / inputs['input_ids'].size(1)).item()
        return perplexity
    
    def run(self, storage: DataFlowStorage, input_key: str = 'raw_content', output_key: str = 'PerplexityScore'):
        # Read the data, evaluate the score, and save the results
        dataframe = storage.read("dataframe")
        self.logger.info(f"Perplexity score ready to evaluate.")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores      
        storage.write(dataframe)
