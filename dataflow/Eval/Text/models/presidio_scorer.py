from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from transformers import AutoModelForTokenClassification, AutoTokenizer
import warnings

# Presidio PII detection Scorer with device support
@MODEL_REGISTRY.register()
class PresidioScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.language = args_dict.get('language', 'en')
        self.device = args_dict.get('device', 'cpu')
        self.model_cache_dir = args_dict.get('model_cache_dir') 
        model_name = 'dslim/bert-base-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=self.model_cache_dir).to(self.device)
        
        warnings.filterwarnings("ignore", category=UserWarning, module="spacy_huggingface_pipelines")
        model_config = [{
            "lang_code": self.language,
            "model_name": {
                "spacy": "en_core_web_sm",
                "transformers": model_name
            }
        }]
        
        self.nlp_engine = TransformersNlpEngine(models=model_config)
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine)
        
        self.batch_size = 1
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'PresidioScore'

    def evaluate_batch(self, batch):
        input_texts = next(iter(batch.values()))

        results = []
        for text in input_texts:
            analysis_results = self.analyzer.analyze(text=text, language=self.language)
            pii_count = len(analysis_results)
            results.append(pii_count)
        
        return results
