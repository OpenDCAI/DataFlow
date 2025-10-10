from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class EmbeddingGenerator(OperatorABC):
    '''
    Embedding Generator is a class that generates answers for given input text.
    '''
    def __init__(self, 
                embedding_serving: LLMServingABC, 
                ):
        self.logger = get_logger()
        self.embedding_serving = embedding_serving

        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.logger.info(f"{self.__class__.__name__} initialized.")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "EmbeddingGenerator算子用于从输入文本生成向量表示（embedding），"
                "通常用于语义检索、聚类或下游模型输入等任务。\n\n"
                "输入参数：\n"
                "- embedding_serving：Embedding服务对象，需实现LLMServingABC接口，用于生成文本的向量表示\n"
                "- input_key：输入文本字段名，默认为'text'\n"
                "- output_key：输出向量字段名，默认为'embeddings'\n\n"
                "输出参数：\n"
                "- 包含文本向量的DataFrame，每行对应一个输入文本的embedding\n"
                "- 返回输出字段名（如'embeddings'），可供后续算子引用"
            )
        elif lang == "en":
            return (
                "The EmbeddingGenerator operator generates vector representations (embeddings) "
                "from input text, typically used for semantic retrieval, clustering, or downstream model inputs.\n\n"
                "Input Parameters:\n"
                "- embedding_serving: Embedding service object implementing the LLMServingABC interface for generating text embeddings\n"
                "- input_key: Field name for input text, default is 'text'\n"
                "- output_key: Field name for output embeddings, default is 'embeddings'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing text embeddings, where each row corresponds to one input text\n"
                "- Returns the output field name (e.g., 'embeddings') for subsequent operator reference"
            )
        else:
            return (
                "EmbeddingGenerator generates vector embeddings from text input for retrieval or representation learning tasks."
            )


    def run(self,
            storage: DataFlowStorage,
            input_key: str = "text",
            output_key: str = "embeddings",
            ):
        dataframe = storage.read("dataframe")
        self.input_key = input_key
        self.output_key = output_key

        texts = dataframe[self.input_key].tolist()
        embeddings_list = self.embedding_serving.generate_embedding_from_input(texts)
        # embeddings = torch.tensor(embeddings_list)

        dataframe[self.output_key] = embeddings_list

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]
