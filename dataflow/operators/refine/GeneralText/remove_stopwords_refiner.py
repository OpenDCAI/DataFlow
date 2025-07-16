import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class RemoveStopwordsRefiner(OperatorABC):
    def __init__(self, model_cache_dir: str = './dataflow_cache'):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.model_cache_dir = model_cache_dir
        nltk.data.path.append(self.model_cache_dir)
        nltk.download('stopwords', download_dir=self.model_cache_dir)
    
    def remove_stopwords(self, text):
        words = text.split()
        stopwords_list = set(stopwords.words('english'))
        refined_words = [word for word in words if word.lower() not in stopwords_list]
        return " ".join(refined_words)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "移除文本中的停用词（如“the”，“is”）" if lang == "zh" else "Remove stopwords from text (e.g., 'the', 'is')."
    
    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            refined_text = self.remove_stopwords(original_text)

            if original_text != refined_text:
                item = refined_text
                modified = True
                self.logger.debug(f"Modified text for key '{self.input_key}': Original: {original_text[:30]}... -> Refined: {refined_text[:30]}...")

            refined_data.append(item)
            if modified:
                numbers += 1
                self.logger.debug(f"Item modified, total modified so far: {numbers}")
        self.logger.info(f"Refining Complete. Total modified items: {numbers}")
        dataframe[self.input_key] = refined_data
        output_file = storage.write(dataframe)
        return [self.input_key]