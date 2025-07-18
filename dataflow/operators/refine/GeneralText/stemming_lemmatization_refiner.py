import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class StemmingLemmatizationRefiner(OperatorABC):
    def __init__(self, method: str = "stemming"):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.method = method.lower()
        if self.method not in ["stemming", "lemmatization"]:
            raise ValueError("Invalid method. Choose 'stemming' or 'lemmatization'.")
        
        nltk.download('wordnet') 
        nltk.download('omw-1.4')  

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对文本进行词干提取或词形还原处理，将词语转换为其基本形式。\n"
                "支持两种处理方式：Porter词干提取(stemming)和WordNet词形还原(lemmatization)，可通过参数选择。\n"
                "输入参数：\n"
                "- method：处理方法，可选'stemming'或'lemmatization'，默认为'stemming'\n"
                "运行参数：\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 处理后的DataFrame，包含词干/词形还原后的文本\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator applies stemming or lemmatization to text, converting words to their base forms.\n"
                "Supports two processing methods: Porter stemming and WordNet lemmatization, selectable via parameter.\n"
                "Input Parameters:\n"
                "- method: Processing method, optional 'stemming' or 'lemmatization', default is 'stemming'\n"
                "Runtime Parameters:\n"
                "- input_key: Input text field name\n"
                "Output Parameters:\n"
                "- Processed DataFrame containing text with stemming/lemmatization applied\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return "Applies stemming or lemmatization to text using NLTK's PorterStemmer or WordNetLemmatizer."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key}...")
        dataframe = storage.read("dataframe")
        numbers = 0
        refined_data = []
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        for item in tqdm(dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"):
            modified = False
            original_text = item
            
            if self.method == "stemming":
                refined_text = " ".join([stemmer.stem(word) for word in original_text.split()])
            elif self.method == "lemmatization":
                refined_text = " ".join([lemmatizer.lemmatize(word) for word in original_text.split()])

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