from dataflow.core import TextRefiner
from dataflow.data import TextDataset
import spacy
from dataflow.utils.registry import PROCESSOR_REGISTRY
from tqdm import tqdm


"""
The NERRefiner class is a text refiner that uses Named Entity Recognition (NER) to identify and mask specific entities in text fields.
Using spaCy’s language model, this class scans each specified field within a dataset for named entities, such as names, organizations, locations, dates, etc.
When an entity is detected (e.g., a person's name or location), it is replaced by a label corresponding to the entity type (e.g., "[PERSON]", "[ORG]").
This modified dataset is returned along with a count of the modified items, allowing sensitive information to be masked or anonymized effectively.
"""

@PROCESSOR_REGISTRY.register()
class NERRefiner(TextRefiner):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.refiner_name = 'NERRefiner'
        self.nlp = spacy.load("en_core_web_sm")


    def refine_func(self, dataset):
        refined_data = []
        numbers = 0
        keys = dataset.keys if isinstance(dataset.keys, list) else [dataset.keys]

        for item in tqdm(dataset, desc=f"Implementing {self.refiner_name}"):
            if isinstance(item, dict):
                modified = False
                for key in keys:
                    if key in item and isinstance(item[key], str):
                        original_text = item[key]
                        refined_text = original_text

                        doc = self.nlp(refined_text)
                        for ent in doc.ents:
                            if ent.label_ in ENTITY_LABELS :
                                refined_text = refined_text.replace(ent.text, f"[{ent.label_}]")

                        if original_text != refined_text:
                            item[key] = refined_text
                            modified = True

                refined_data.append(item)
                if modified:
                    numbers += 1
        dataset.dataset = refined_data
        return dataset, numbers



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
