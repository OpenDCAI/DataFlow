import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

@OPERATOR_REGISTRY.register()
class KeywordExtractor(OperatorABC):
    """
    KeywordExtractor 使用大语言模型从文本中提取主题关键词。
    """
    def __init__(self, llm_serving: LLMServingABC, num_keywords: int = 5, lang: str = "en"):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.num_keywords = num_keywords
        self.lang = lang

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "关键词提取算子：调用LLM为文本片段提取指定数量的主题关键词。\n\n"
                "输入参数：\n- input_key: 文本字段名\n- output_key: 关键词字段名\n- num_keywords: 关键词数量\n" )
        else:
            return (
                "Keyword extraction operator: uses an LLM to extract thematic keywords from text segments.\n\n"
                "Input Parameters:\n- input_key: field containing the text\n- output_key: field to save extracted keywords\n- num_keywords: number of keywords to extract\n" )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str, output_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")
        if output_key in df.columns:
            raise ValueError(f"Column {output_key} already exists and would be overwritten")

    def _build_prompts(self, df: pd.DataFrame, input_key: str) -> list:
        tpl_en = (
            "Extract {k} thematic keywords (single words or short phrases) from the following text. "
            "Return them as a comma-separated list without additional explanation.\n\nText:\n" )
        tpl_zh = (
            "请从以下文本中提取 {k} 个主题关键词（单词或短语），仅以逗号分隔的列表返回，不要额外说明。\n\n文本：\n" )
        tpl = tpl_zh if self.lang == "zh" else tpl_en
        return [tpl.format(k=self.num_keywords) + row[input_key] for _, row in df.iterrows()]

    def _parse_response(self, resp: str) -> str:
        return resp.strip().replace("\n", " ")

    def run(self, storage: DataFlowStorage, input_key: str = "instruction", output_key: str = "keywords", num_keywords: int = None):
        # 如果调用方传入了 num_keywords，以调用方为准
        if num_keywords is not None:
            self.num_keywords = num_keywords

        # 读取数据
        df = storage.read("dataframe")

        # 当默认 input_key 不存在时，尝试在常见字段名中寻找可用的列
        if input_key not in df.columns:
            fallback_keys = ["instruction", "input", "text", "source"]
            for alt_key in fallback_keys:
                if alt_key in df.columns:
                    self.logger.warning(
                        f"Input column '{input_key}' not found, automatically switched to '{alt_key}'.")
                    input_key = alt_key
                    break

        # 校验
        self._validate_dataframe(df, input_key, output_key)

        # 构造 LLM prompt 并获取响应
        prompts = self._build_prompts(df, input_key)
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="")

        # 解析并写回
        df[output_key] = [self._parse_response(r) for r in responses]
        output_file = storage.write(df)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="../example_data/DataflowAgent/agent_test_data.json",
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
operator = KeywordExtractor(llm_serving=llm_serving, num_keywords=5, lang='en')

# 4. Run
operator.run(storage=storage.step())
