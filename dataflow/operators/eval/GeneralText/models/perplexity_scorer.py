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
        return "基于 Kenlm 模型计算文本的困惑度，困惑度越低，文本的流畅性和可理解性越高。" if lang == "zh" else "Uses Kenlm to calculate perplexity; lower scores indicate better fluency."

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


