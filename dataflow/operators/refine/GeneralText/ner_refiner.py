import spacy
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

ENTITY_LABELS = {
    "PERSON": "[PERSON]", 
    "ORG": "[ORG]",  
    "GPE": "[GPE]",  
    "LOC": "[LOC]",  
    "PRODUCT": "[PRODUCT]",  
    "EVENT": "[EVENT]", 
    "DATE": "[DATE]",  
    "TIME": "[TIME]",  
    "MONEY": "[MONEY]", 
    "PERCENT": "[PERCENT]",  
    "QUANTITY": "[QUANTITY]",  
    "ORDINAL": "[ORDINAL]",  
    "CARDINAL": "[CARDINAL]",  
    "NORP": "[NORP]",  
    "FAC": "[FAC]",  
    "LAW": "[LAW]",  
    "LANGUAGE": "[LANGUAGE]",  
    "WORK_OF_ART": "[WORK_OF_ART]",  
    "LAW": "[LAW]",  
    "ORDINAL": "[ORDINAL]",  
    "CARDINAL": "[CARDINAL]", 
    "PERCENT": "[PERCENT]", 
    "QUANTITY": "[QUANTITY]",  
    "DATE": "[DATE]",  
    "TIME": "[TIME]",  
    "URL": "[URL]",  
    "EMAIL": "[EMAIL]",  
    "MONEY": "[MONEY]",  
    "FAC": "[FAC]",  
    "PRODUCT": "[PRODUCT]",  
    "EVENT": "[EVENT]",  
    "WORK_OF_ART": "[WORK_OF_ART]",  
    "LANGUAGE": "[LANGUAGE]",  
    "NORP": "[NORP]"  
}

@OPERATOR_REGISTRY.register()
class NERRefiner(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.nlp = spacy.load("en_core_web_sm")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用命名实体识别（NER）技术识别并屏蔽文本中的特定实体" if lang == "zh" else "Mask specific entities in text using Named Entity Recognition (NER)."
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False  
            original_text = item
            refined_text = original_text

            doc = self.nlp(refined_text)
            for ent in doc.ents:
                if ent.label_ in ENTITY_LABELS :
                    refined_text = refined_text.replace(ent.text, f"[{ent.label_}]")

            if original_text != refined_text:
                item = refined_text
                modified = True

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]