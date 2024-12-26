from dataflow.core import TextRefiner
from dataflow.utils.registry import PROCESSOR_REGISTRY
from transformers import AutoModelForTokenClassification, AutoTokenizer
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import warnings
from tqdm import tqdm

"""
The PIIAnonymizeRefiner class is a text refiner designed to anonymize Personally Identifiable Information (PII) in specified text fields within a dataset.
It uses both a BERT-based model ('dslim/bert-base-NER') and spaCy for Named Entity Recognition (NER) to identify sensitive information, such as names, locations, and other personal data.
Once detected, the `Presidio` library anonymizes this PII by replacing it with placeholders (e.g., "[PERSON]" for names). 

The refiner initializes a tokenizer and model for entity recognition and configures a custom NLP engine for PII analysis. 
During processing, the original text is analyzed, PII is anonymized, and the modified dataset is returned, along with a count of anonymized items. 
This class is especially useful for datasets that need de-identification for privacy compliance.
"""

@PROCESSOR_REGISTRY.register()
class PIIAnonymizeRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'PIIAnonymizeRefiner'

        self.language = args_dict.get('language')
        self.device = args_dict.get('device')
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
        self.anonymizer = AnonymizerEngine()

    def refine_func(self, dataset):
        anonymized_count = 0
        keys = dataset.keys if isinstance(dataset.keys, list) else [dataset.keys]
        
        for item in tqdm(dataset, desc=f"Implementing {self.refiner_name}"):
            if isinstance(item, dict):
                modified = False
                for key in keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        results = self.analyzer.analyze(original_text, language=self.language)
                        anonymized_text = self.anonymizer.anonymize(original_text, results)
                        if original_text != anonymized_text.text:
                            item[key] = anonymized_text.text
                            modified = True

                if modified:
                    anonymized_count += 1
        return dataset, anonymized_count
