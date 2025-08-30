from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow import get_logger
import pandas as pd

@OPERATOR_REGISTRY.register()
class ParaphraseAugmentOperator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对指定字段文本进行同义改写，从而在不破坏原始数据完整性的前提下增加训练样本的多样性。\n\n"
                "输入参数：\n"
                "- input_key: 需要进行同义改写的字段名（默认 \"question\"）\n"
                "- output_key: 新增的同义改写字段名（默认 \"paraphrased_question\"）\n"
            )
        else:
            return (
                "This operator paraphrases the text in the specified column to increase training data diversity while keeping the original data intact.\n\n"
                "Input Parameters:\n"
                "- input_key: Name of the column to paraphrase (default \"question\")\n"
                "- output_key: Name of the new column that will store the paraphrased text (default \"paraphrased_question\")\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required = [self.input_key]
        forbidden = [self.output_key]
        miss = [k for k in required if k not in dataframe.columns]
        conflict = [k for k in forbidden if k in dataframe.columns]
        if miss:
            raise ValueError(f"Missing required column(s): {miss}")
        if conflict:
            raise ValueError(f"Column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, texts):
        system_prompt = "You are an expert rewriter. Given a question, create one high-quality paraphrase that keeps the original meaning but uses different wording. Only return the paraphrased question without any extra text."
        user_prompts = [f"Original question: {t}\nParaphrased:" for t in texts]
        return system_prompt, user_prompts

    def run(self,
            storage: DataFlowStorage,
            input_key: str = "question",
            output_key: str = "paraphrased_question"):
        self.input_key, self.output_key = input_key, output_key
        df = storage.read("dataframe")
        self._validate_dataframe(df)

        system_prompt, user_prompts = self._reformat_prompt(df[self.input_key].tolist())
        paraphrases = self.llm_serving.generate_from_input(user_prompts, system_prompt)
        df[self.output_key] = paraphrases

        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return self.output_key


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
        api_url="https://api.chatanywhere.com.cn/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o-mini",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = ParaphraseAugmentOperator(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
