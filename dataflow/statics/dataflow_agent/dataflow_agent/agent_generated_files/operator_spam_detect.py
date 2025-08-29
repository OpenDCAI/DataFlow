from typing import Optional

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage
import numpy as np
from tqdm import tqdm


@OPERATOR_REGISTRY.register()
class SpamWaterContentFilter(OperatorABC):
    def __init__(self, llm_serving, threshold: float = 0.5, batch_size: int = 16):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.threshold = threshold
        self.batch_size = batch_size
        self.logger.info(
            f"Initializing {self.__class__.__name__} with threshold = {self.threshold}, batch_size = {self.batch_size}."
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        return (
            "使用大模型检测并过滤垃圾短信和水内容"
            if lang == "zh"
            else "Detect and filter spam or low-value (water) content using an LLM."
        )

    def _batched(self, iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    def _resolve_input_key(self, df, provided_key: Optional[str]):
        """Return a valid column name to run the filter on."""
        if provided_key and provided_key in df.columns:
            return provided_key
        for candidate in ("output", "instruction"):
            if candidate in df.columns:
                return candidate
        # Fall back to the first column if nothing matches.
        return df.columns[0]

    # unified llm call to be compatible with different llm_serving implementations
    def _generate(self, prompts, **kwargs):
        """Call LLM serving with best-effort method detection."""
        # Priority 1: standard generate
        if hasattr(self.llm_serving, "generate"):
            return self.llm_serving.generate(prompts, **kwargs)
        # Priority 2: batch generate
        if hasattr(self.llm_serving, "generate_batch"):
            return self.llm_serving.generate_batch(prompts, **kwargs)
        # Priority 3: generate_from_input (used by APILLMServing_request)
        if hasattr(self.llm_serving, "generate_from_input"):
            # extract system_prompt if provided, otherwise use default
            system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant")
            return self.llm_serving.generate_from_input(prompts, system_prompt=system_prompt)
        # Priority 4: llm_serving itself is callable
        if callable(self.llm_serving):
            return self.llm_serving(prompts, **kwargs)
        raise AttributeError(
            "The provided llm_serving instance does not have a compatible generate interface."
        )

    # Added default value for input_key so the framework can call run(storage) directly.
    def run(
        self,
        storage: DataFlowStorage,
        input_key: Optional[str] = None,
        output_key: str = "spam_water_filter_label",
    ):
        df = storage.read("dataframe")
        self.input_key = self._resolve_input_key(df, input_key)
        self.output_key = output_key

        self.logger.info(
            f"Running {self.__class__.__name__} on column '{self.input_key}' …"
        )

        texts = df[self.input_key].fillna("").tolist()
        valid = []
        for batch in tqdm(
            list(self._batched(texts, self.batch_size)),
            desc=f"Implementing {self.__class__.__name__}",
        ):
            prompts = [
                "You are a content quality inspector. Reply with a single number between 0 and 1 where 1 means the text is spam or meaningless filler (water content) and 0 means the text is useful and clean. Text:" + t
                for t in batch
            ]
            # use the unified _generate method
            responses = self._generate(prompts)
            # ensure responses is list-like
            if not isinstance(responses, (list, tuple)):
                responses = [responses]
            for r in responses:
                # some llm servers may return dict/object with "text" attr
                if not isinstance(r, str):
                    r = getattr(r, "text", str(r))
                try:
                    score = float(r.strip())
                except ValueError:
                    score = 1.0  # Treat unparsable responses as spam/high score
                valid.append(score < self.threshold)

        valid = np.array(valid, dtype=int)
        df[self.output_key] = valid
        filtered_df = df[valid == 1]
        storage.write(filtered_df)
        self.logger.info(
            f"Filtering completed. Total records passing filter: {len(filtered_df)}."
        )
        return [self.output_key]


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
operator = SpamWaterContentFilter(llm_serving=llm_serving, threshold=0.5, batch_size=16)

# 4. Run
operator.run(storage=storage.step())
