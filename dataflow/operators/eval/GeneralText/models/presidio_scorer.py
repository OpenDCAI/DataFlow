from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage

# Presidio PII detection Scorer
@OPERATOR_REGISTRY.register()
class PresidioScorer(OperatorABC):
    def __init__(self, device='cuda', lang='en', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.language = lang
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.model_name = 'dslim/bert-base-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        warnings.filterwarnings("ignore", category=UserWarning, module="spacy_huggingface_pipelines")
        model_config = [{
            "lang_code": self.language,
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": self.model_name
            }
        }]
        self.nlp_engine = TransformersNlpEngine(models=model_config)
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine)
        self.score_name = 'PresidioScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用Microsoft Presidio模型识别文本中的私人实体，返回PII信息个数。" if lang == "zh" else "Detects PII entities using Microsoft Presidio; returns number of detected items."

    def eval(self, dataframe, input_key):
        input_texts = dataframe.get(input_key, '').to_list()
        results = []
        self.logger.info(f"Evaluating {self.score_name}...")
        for text in input_texts:
            analysis_results = self.analyzer.analyze(text=text, language=self.language)
            pii_count = len(analysis_results)
            results.append(pii_count)
        self.logger.info("Evaluation complete!")
        return results
    
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='PresidioScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)        
        dataframe[output_key] = scores
        storage.write(dataframe)
