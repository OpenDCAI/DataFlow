import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.serving.light_rag_serving import LightRAGServing

@OPERATOR_REGISTRY.register()
class RetrievalGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LightRAGServing,
                 system_prompt: str = "You are a helpful agent.",
                 ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt

    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "根据知识库内容和输入文本生成回答。\n\n"
                "功能说明：\n"
                "- 利用LightRAG提取的知识库来辅助生成回答。"
                "输入要求：\n\n"
                "__init__函数参数：\n"
                "- llm_serving: LightRAGServing实例，需要先创建实例并导入文件\n"
                "- system_prompt: 系统prompt(默认：'You are a helpful agent.')\n"
                "run输入参数：\n"
                "- storage: 存储文本、LLM回答的DataFlowStorage实例\n"
                "- input_key: 提供的文本内容列名(默认：'raw_content')\n"
                "- output_key: 生成的回答列名(默认：'generated_content')\n"
                "输出：在storage中添加的回答列名\n"
            )
        elif lang == "en":
            return (
                "Generate answers based on the knowledge base and the input text.\n\n"
                "Function Description:\n"
                "- Use the knowledge base extracted by LightRAG to assist in generating answers."
                "Input Requirements:\n\n"
                "Parameters of the __init__ function:\n"
                "- llm_serving: A LightRAGServing instance, which must be created and loaded with files beforehand.\n"
                "- system_prompt: The system prompt (default: 'You are a helpful agent.')\n"
                "Parameters for the run method:\n"
                "- storage: A DataFlowStorage instance that stores the input text and the LLM responses\n"
                "- input_key: The column name containing the provided text (default: 'raw_content')\n"
                "- output_key: The column name for generated answers (default: 'generated_content')\n"
                "Output: The name of the column added to storage that contains the generated answers.\n"
            )
        else:
            return "QAGenerator generates QA pairs for given document fragments."

    
    async def run(self,
            storage: DataFlowStorage,
            input_key: str = "raw_content",
            output_key: str = "generated_content",
            ):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running RetrievalGenerator...")

        # Load the raw dataframe from the input file
        df = storage.read('dataframe')
        self.logger.info(f"Loading, number of tasks: {len(df)}")

        llm_inputs = []
        for index, row in df.iterrows():
            raw_content = row.get(self.input_key, '')
            if raw_content:
                llm_input = str(raw_content)
                llm_inputs.append(llm_input)

        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = await self.llm_serving.generate_from_input(llm_inputs, self.system_prompt)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return
        
        df[self.output_key] = generated_outputs

        output_file = storage.write(df)
        return output_key