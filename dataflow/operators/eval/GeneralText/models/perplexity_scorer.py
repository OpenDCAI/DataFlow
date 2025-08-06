from dataflow.core import OperatorABC
from dataflow.operators.eval.GeneralText.models.Kenlm.model import KenlmModel
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.utils import get_logger
# Kenlm models perplexity evaluation
@OPERATOR_REGISTRY.register()
class PerplexityScorer(OperatorABC):
    # Need to download model first!
    def __init__(self, lang='en', model_name='dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = model_name
        self.language = lang
        self.score_name = 'PerplexityScore'
        try:
            self.model = KenlmModel.from_pretrained(self.model_name, self.language)
            self.logger.info(f'{self.__class__.__name__} initialized.')
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.error("The model has not been downloaded yet.")
            self.logger.error("Please download the model from: https://huggingface.co/edugp/kenlm/tree/main")
            raise RuntimeError(f"Model loading failed. Please download the model from the provided link: https://huggingface.co/edugp/kenlm/tree/main. For default configuration, you can download en.arpa.bin, en.sp.model and en.sp.vocab, and put them in the folder dataflow/operators/GeneralText/models/Kenlm/wikipedia")
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于Kenlm语言模型计算文本的困惑度(Perplexity)，困惑度越低表示文本的流畅性和可理解性越高。支持多语言模型，" 
                "默认使用维基百科训练的英文模型。需要先从HuggingFace下载模型文件(en.arpa.bin, en.sp.model和en.sp.vocab)。\n" 
                "输入参数：\n" 
                "- text: 待评估的文本字符串\n" 
                "- lang: 语言类型，默认为'en'\n" 
                "- model_name: 模型路径，默认为'dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia'\n" 
                "输出参数：\n" 
                "- float: 困惑度值，越低表示文本流畅性越好"
            )
        else:
            return (
                "Calculate text perplexity using the Kenlm language model. Lower perplexity indicates better fluency and comprehensibility. Supports multilingual models, " 
                "defaulting to an English model trained on Wikipedia. Requires downloading model files (en.arpa.bin, en.sp.model, and en.sp.vocab) from HuggingFace first.\n" 
                "Input parameters:\n" 
                "- text: Text string to be evaluated\n" 
                "- lang: Language type, default 'en'\n" 
                "- model_name: Model path, default 'dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia'\n" 
                "Output parameters:\n" 
                "- float: Perplexity value, lower values indicate better text fluency"
            )

    def eval(self, dataframe, input_key):
        input_texts = dataframe.get(input_key, '').to_list()
        self.logger.info(f"Evaluating {self.score_name}...")
        results = []
        for text in input_texts:
            perplexity = self.model.get_perplexity(text)
            results.append(perplexity)
        self.logger.info("Evaluation complete!")
        return results
    
    def run(self, storage: DataFlowStorage, input_key: str = 'raw_content', output_key: str = 'PerplexityScore'):
        # Read the dataframe, evaluate scores, and store results
        dataframe = storage.read("dataframe")
        self.logger.info(f"Perplexity score ready to evaluate.")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores      
        storage.write(dataframe)


