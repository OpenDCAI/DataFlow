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
        if lang == "zh":
            return (
                "该算子用于移除文本中的英语停用词（如\"the\"，\"is\"，\"in\"等无实际意义的高频词汇）。\n"
                "使用NLTK库的stopwords语料库进行停用词过滤，提高文本特征密度。\n"
                "输入参数：\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含去除停用词的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator removes English stopwords from text (e.g., high-frequency words with little meaning like \"the\", \"is\", \"in\").\n"
                "Uses NLTK library's stopwords corpus for stopword filtering to improve text feature density.\n"
                "Input Parameters:\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with stopwords removed\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Removes English stopwords from text using NLTK's stopwords corpus."
    
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