from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import pandas as pd

@OPERATOR_REGISTRY.register()
class MedicalQuestionParaphraser(OperatorABC):
    """
    Generate paraphrased versions of medical scenario questions and append them as a
    new column ("questionPARA") to the existing dataframe rows.
    """

    def __init__(self,
                 llm_serving: LLMServingABC,
                 n_paraphrases: int = 1,
                 system_prompt: str = (
                     "You are a medical question rewriting expert. For the given medical scenario "
                     "question, provide a concise, semantically equivalent paraphrase."),
                 separator: str = "\n"):  # used when n_paraphrases > 1
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.n_paraphrases = max(1, min(n_paraphrases, 5))  # clamp to 1-5
        self.system_prompt = system_prompt.strip()
        self.separator = separator

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "为医疗场景问题生成释义版本（paraphrase），并将结果写入新列 questionPARA。\n"
                "输入参数：\n"
                "- input_key: 原问题所在列名，默认 'question'\n"
                "- output_key: 释义结果列名，固定为 'questionPARA'\n"
                "- n_paraphrases: 每行生成多少个释义（1-5），默认 1\n"
                "输出：新增列 questionPARA，内容为 LLM 生成的释义（若 n_paraphrases>1 用换行分隔）"
            )
        else:
            return (
                "Generate paraphrased medical scenario questions and add them to a new column "
                "'questionPARA'.\nInput Parameters:\n- input_key: column containing original questions (default 'question')\n- n_paraphrases: number of paraphrases per question (1-5, default 1)\nOutput: dataframe with new column 'questionPARA' holding the generated paraphrases."
            )

    def _validate(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Required column '{input_key}' is missing in the dataframe.")
        if output_key in df.columns:
            self.logger.warning(
                f"Column '{output_key}' already exists and will be overwritten.")

    def _build_prompt(self, question: str) -> str:
        if self.n_paraphrases == 1:
            return (f"{self.system_prompt}\n\n"
                    f"Original question: {question}\n"
                    f"Paraphrased version:")
        else:
            return (f"{self.system_prompt}\n\n"
                    f"Original question: {question}\n"
                    f"Provide {self.n_paraphrases} paraphrased versions, each on a new line:")

    def run(self, storage: DataFlowStorage, input_key: str = "question", output_key: str = "questionPARA"):
        self.logger.info("Running MedicalQuestionParaphraser …")
        df = storage.read("dataframe")
        self._validate(df, input_key, output_key)

        prompts = [self._build_prompt(q) if isinstance(q, str) and q.strip() else "" for q in df[input_key]]

        # Filter out empty prompts and remember their indices
        idx_and_prompts = [(idx, p) for idx, p in enumerate(prompts) if p]
        if not idx_and_prompts:
            self.logger.warning("No valid questions to paraphrase.")
            output_file = storage.write(df)
            return [output_key]

        indices, valid_prompts = zip(*idx_and_prompts)
        try:
            responses = self.llm_serving.generate_from_input(list(valid_prompts))
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise e

        # Post-process: ensure string output, join list if model returned list
        processed = []
        for resp in responses:
            if isinstance(resp, list):
                processed.append(self.separator.join(map(str, resp)))
            else:
                processed.append(str(resp).strip())

        # Insert results back to dataframe
        df[output_key] = ""  # init column
        for i, paraphrase in zip(indices, processed):
            df.at[i, output_key] = paraphrase

        output_file = storage.write(df)
        self.logger.info(f"Paraphrased questions saved to {output_file}")
        return [output_key]


# ======== Auto-generated runner ========
# from dataflow.utils.storage import FileStorage
# from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
# from dataflow.core import LLMServingABC

# if __name__ == "__main__":
#     # 1. FileStorage
#     storage = FileStorage(
#         first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl",
#         cache_path="./cache_local",
#         file_name_prefix="dataflow_cache_step",
#         cache_type="jsonl",
#     )

#     # 2. LLM-Serving
#     # -------- LLM Serving (Remote) --------
#     llm_serving = APILLMServing_request(
#         api_url="http://123.129.219.111:3000/v1/chat/completions",
#         key_name_of_api_key = 'DF_API_KEY',
#         model_name="gpt-4o",
#         max_workers=100,
#     )
#     # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# # 3. Instantiate operator
# operator = MedicalQuestionParaphraser(llm_serving=llm_serving, n_paraphrases=1, system_prompt='You are a medical question rewriting expert. For the given medical scenario question, provide a concise, semantically equivalent paraphrase.', separator='\n')

# # 4. Run
# operator.run(storage=storage.step())
