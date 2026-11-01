import fasttext
import numpy as np
from huggingface_hub import hf_hub_download
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from tqdm import tqdm
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class LanguageFilter(OperatorABC):

    def __init__(self, allowed_languages: list, model_cache_dir: str = None):
        self.logger = get_logger()
        self.filter_name = 'LanguageFilter'
        self.logger.info(f"Initializing {self.__class__.__name__} with allowed_languages = {allowed_languages} and model_cache_dir = {model_cache_dir}...")
        
        self.allowed_languages = allowed_languages
        self.model_cache_dir = model_cache_dir
        
        # Download and load the FastText language model
        try:
            self.logger.info("Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin", cache_dir=self.model_cache_dir)
            self.model = fasttext.load_model(model_path)
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error downloading or loading model: {e}")
            raise

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用FastText语言识别模型过滤数据。下载并加载预训练的FastText语言识别模型，检查文本的语言是否在允许的语言列表中。\n"
                "输入参数：\n"
                "- allowed_languages：允许的语言标签列表\n"
                "- model_cache_dir：模型缓存目录路径\n"
                "- input_key：输入文本字段名（必填）\n"
                "- output_key：写入语言判定标签的输出字段名，默认为'language_label'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留语言在允许列表中的文本（FastText前5个预测标签中任一在allowed_languages中即通过）；"
                "每行新增output_key字段，值为整数标签（1=通过，0=未通过；由于只保留通过的行，输出中该值均为1），不是语言代码，该字段保留在输出中\n"
                "- 返回包含output_key的列表"
            )
        else:
            return (
                "Filter data using FastText language identification model. Downloads and loads pre-trained FastText language identification model to check if text language is in allowed list.\n"
                "Input Parameters:\n"
                "- allowed_languages: List of allowed language labels\n"
                "- model_cache_dir: Model cache directory path\n"
                "- input_key: Input text field name (required)\n"
                "- output_key: Name of the output field that receives the language pass label, default is 'language_label'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with language in allowed list (a row passes if any of FastText's top-5 predicted labels is in allowed_languages); "
                "each row gets a new output_key field holding an integer label (1 = pass, 0 = fail; since only passing rows are kept, it is 1 in the output), not a language code, which is kept in the output\n"
                "- List containing output_key"
            )

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")

        predictions = []

        # Assuming the dataframe contains the text in `input_key`
        for text in tqdm(dataframe[input_key], desc=f"Implementing {self.filter_name}"):
            labels, scores = self.model.predict(text.replace('\n', ' '), k=5)
            label_score_pairs = list(zip(labels, scores))
            label_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            top_labels = [label for label, score in label_score_pairs]
            predictions.append(any(label in self.allowed_languages for label in top_labels))

        return np.array(predictions).astype(int)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='language_label'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name} with input_key = {self.input_key} and output_key = {self.output_key}...")
        predictions = self.eval(dataframe, self.input_key)
        dataframe[self.output_key] = predictions
        filtered_dataframe = dataframe[dataframe[self.output_key] == 1]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
