from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd

@OPERATOR_REGISTRY.register()
class QuestionParaphraser(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC,
                 paraphrase_prompt: str = "Please rewrite the following question in a different wording but keep the same meaning:\n\n{question}"):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.paraphrase_prompt = paraphrase_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "生成与原问题语义一致但表达不同的同义问题，以增加训练数据多样性。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务实例\n"
                "- paraphrase_prompt：可选，自定义改写提示词\n"
                "- input_key：原问题字段名，默认为 'question'\n"
                "- output_key：输出同义问题字段名，默认为 'questionPARA'\n"
                "输出：\n"
                "- 返回包含新列 'questionPARA' 的 DataFrame，并返回该列名供后续算子使用"
            )
        else:
            return (
                "Generate semantically equivalent but differently worded questions to enlarge training data diversity.\n"
                "Input Params:\n"
                "- llm_serving: LLM serving instance\n"
                "- paraphrase_prompt: optional custom prompt template\n"
                "- input_key: column name of original questions, default 'question'\n"
                "- output_key: column name for paraphrased questions, default 'questionPARA'\n"
                "Output: DataFrame with new column 'questionPARA' and the column name for pipeline reference"
            )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column '{input_key}' in input dataframe")
        if output_key in df.columns:
            self.logger.warning(f"Column '{output_key}' already exists and will be overwritten")

    def run(self, storage: DataFlowStorage, input_key: str = "question", output_key: str = "questionPARA"):
        self.logger.info("Running QuestionParaphraser operator …")
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key, output_key)

        prompts = []
        for q in df[input_key]:
            prompt = self.paraphrase_prompt.format(question=q)
            prompts.append(prompt)

        try:
            paraphrased = self.llm_serving.generate_from_input(prompts)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise

        df[output_key] = paraphrased
        output_file = storage.write(df)
        self.logger.info(f"Paraphrased questions saved to {output_file}")
        return output_key


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Remote) --------
    llm_serving = APILLMServing_request(
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = QuestionParaphraser(llm_serving=llm_serving, paraphrase_prompt='Please rewrite the following question in a different wording but keep the same meaning:\n\n{question}')

# 4. Run
operator.run(storage=storage.step())
