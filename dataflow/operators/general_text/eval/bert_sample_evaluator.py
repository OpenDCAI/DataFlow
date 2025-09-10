from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import evaluate

@OPERATOR_REGISTRY.register()
class BertSampleEvaluator(OperatorABC):
    def __init__(self, lang='en', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.data_type = "text"
        self.score_name = "BERTScore"
        self.lang = lang
        self.model_type = "distilbert-base-uncased"
        self.idf = False
        self.rescale_with_baseline = False
        self.bertscore = evaluate.load("bertscore", cache_dir=model_cache_dir)
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用BERTScore评估生成文本与参考文本的相似度，基于上下文嵌入计算P/R/F1分数。\n"
                "输入参数：\n"
                "- lang：语言类型，默认为'en'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- input_key：生成文本字段名\n"
                "- reference_key：参考文本字段名\n"
                "- output_key：输出得分字段名，默认为'BertScore'\n"
                "输出参数：\n"
                "- 包含F1相似度得分的DataFrame"
            )
        elif lang == "en":
            return (
                "Evaluate similarity between generated and reference text using BERTScore with contextual embeddings.\n"
                "Input Parameters:\n"
                "- lang: Language type, default 'en'\n"
                "- model_cache_dir: Model cache directory, default './dataflow_cache'\n"
                "- input_key: Field name for generated text\n"
                "- reference_key: Field name for reference text\n"
                "- output_key: Field name for output score, default 'BertScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing F1 similarity scores"
            )
        else:
            return "Evaluate text similarity using BERTScore."
    
    def eval(self, dataframe, input_key, reference_key):
        eval_data = dataframe[input_key].to_list()
        ref_data = dataframe[reference_key].to_list()
        self.logger.info(f"Evaluating {self.score_name}...")
        if ref_data is None:
            raise ValueError("Reference data must be provided for BERTScorer")
        results = self.bertscore.compute(
            predictions=eval_data,
            references=ref_data,
            lang=self.lang,
            model_type=self.model_type,
            idf=self.idf,
            rescale_with_baseline=self.rescale_with_baseline
        )
        scores = results["f1"]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, reference_key: str, output_key: str='BertScore'):
        self.input_key = input_key
        self.reference_key = reference_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key, reference_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
