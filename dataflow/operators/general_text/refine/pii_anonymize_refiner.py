from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class PIIAnonymizeRefiner(OperatorABC):
    def __init__(self, lang='en', device='cuda', model_cache_dir='./dataflow_cache', model_name='dslim/bert-base-NER', ):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.lang = lang
        self.device = device
        self.model_cache_dir = model_cache_dir
        self.model_name = model_name
        model_name = 'dslim/bert-base-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=self.model_cache_dir).to(self.device)
        model_config = [{
            "lang_code": self.lang,
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": model_name
            }
        }]
        
        self.nlp_engine = TransformersNlpEngine(models=model_config)
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine)
        self.anonymizer = AnonymizerEngine()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用Presidio和BERT-NER模型识别并匿名化文本中的个人身份信息（PII）。支持多种PII类型的检测和匿名化处理。"
                "输入参数：\n"
                "- lang：语言代码，默认为'en'\n"
                "- device：运行设备，默认为'cuda'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- model_name：NER模型名称，默认为'dslim/bert-base-NER'\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含匿名化后文本的DataFrame\n"
                "- 返回输入字段名，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Identify and anonymize Personally Identifiable Information (PII) in text using Presidio and BERT-NER models. Supports detection and anonymization of various PII types.\n"
                "Input Parameters:\n"
                "- lang: Language code, default is 'en'\n"
                "- device: Running device, default is 'cuda'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- model_name: NER model name, default is 'dslim/bert-base-NER'\n"
                "- input_key: Field name for input text\n\n"
                "Output Parameters:\n"
                "- DataFrame containing anonymized text\n"
                "- Returns input field name for subsequent operator reference"
            )
        else:
            return (
                "PIIAnonymizeRefiner identifies and anonymizes PII in text using NLP models."
            )

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        anonymized_count = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False  
            original_text = item
            results = self.analyzer.analyze(original_text, language=self.lang)
            anonymized_text = self.anonymizer.anonymize(original_text, results)
            if original_text != anonymized_text.text:
                item = anonymized_text.text
                modified = True
            self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {anonymized_text.text[:30]}...")

            refined_data.append(item)
            if modified:
                anonymized_count += 1
                self.logger.debug(f"Item modified, total modified so far: {anonymized_count}")
        self.logger.info(f"Refining Complete. Total modified items: {anonymized_count}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]