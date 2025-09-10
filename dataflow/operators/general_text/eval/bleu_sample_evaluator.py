from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.operators.general_text.eval.bleu.bleu import Bleu
from tqdm import tqdm

@OPERATOR_REGISTRY.register()
class BleuSampleEvaluator(OperatorABC):
    def __init__(self, n=4, eff="average", special_reflen=None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'BleuScore'
        valid_eff_options = ["shortest", "average", "longest"]
        if eff not in valid_eff_options:
            raise ValueError(f"Invalid value for 'eff'. Must be one of {valid_eff_options}, but got '{eff}'.")
        self.n = n  # Max n-gram length (default: 4)
        self.eff = eff  # [shortest, average, longest]
        self.special_reflen = special_reflen  # Special reference length if specified
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "计算BLEU分数评估生成文本与参考文本的n-gram重叠度，支持1-4元语法分析。\n"
                "输入参数：\n"
                "- n：最大n-gram长度，默认为4\n"
                "- eff：参考长度计算方式，可选'shortest'/'average'/'longest'，默认为'average'\n"
                "- special_reflen：特殊参考长度，默认为None\n"
                "- input_key：生成文本字段名\n"
                "- reference_key：参考文本字段名\n"
                "- output_key：输出得分字段名，默认为'BleuScore'\n"
                "输出参数：\n"
                "- 包含BLEU得分的DataFrame"
            )
        elif lang == "en":
            return (
                "Evaluate n-gram overlap between generated and reference text using BLEU score (1-4 grams supported).\n"
                "Input Parameters:\n"
                "- n: Maximum n-gram length, default 4\n"
                "- eff: Reference length calculation method, 'shortest'/'average'/'longest', default 'average'\n"
                "- special_reflen: Special reference length, default None\n"
                "- input_key: Field name for generated text\n"
                "- reference_key: Field name for reference text\n"
                "- output_key: Field name for output score, default 'BleuScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing BLEU scores"
            )
        else:
            return "Evaluate text similarity using BLEU score."
    
    def _score_func(self, eval_text, ref_text):
        bleu_scorer = Bleu(
            test=eval_text,
            refs=[ref_text],
            n=self.n,
            special_reflen=self.special_reflen,
        )
        bleu_score, _ = bleu_scorer.compute_score(option=self.eff)
        return bleu_score[0]
    
    def eval(self, dataframe, input_key, reference_key):
        eval_data = dataframe[input_key]
        ref_data = dataframe[reference_key]
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = [self._score_func(eval_text, ref_text) for eval_text, ref_text in tqdm(zip(eval_data, ref_data), desc="BleuScorer Evaluating...")]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, reference_key: str, output_key: str='BleuScore'):
        self.input_key = input_key
        self.reference_key = reference_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")        
        scores = self.eval(dataframe, input_key, reference_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
