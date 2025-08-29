from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import numpy as np
import re


@OPERATOR_REGISTRY.register()
class TextDensityStats(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return (
            "计算文本的多种长度与密度统计信息，并将结果写入数据框"
            if lang == "zh"
            else "Compute various length and density statistics of the text and write the results back to the dataframe."
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "instruction",
        output_prefix: str = "text_density",
    ):
        self.input_key = input_key
        self.output_prefix = output_prefix
        dataframe = storage.read("dataframe")
        self.logger.info(
            f"Running {self.__class__.__name__} with input_key = {self.input_key} ..."
        )

        SENT_PATTERN = re.compile(r"[^.!?\n]+[.!?]*", flags=re.UNICODE)

        word_counts = []
        sentence_counts = []
        char_counts = []
        mean_word_lengths = []
        words_per_sentence = []

        for text in tqdm(
            dataframe[self.input_key], desc=f"Implementing {self.__class__.__name__}"
        ):
            if isinstance(text, str) and text.strip():
                words = text.split()
                wc = len(words)
                sc = len(SENT_PATTERN.findall(text))
                cc = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
                mwl = round(sum(len(w) for w in words) / wc, 2) if wc else 0.0
                wps = round(wc / sc, 2) if sc else 0.0
            else:
                wc = sc = cc = 0
                mwl = wps = 0.0
            word_counts.append(wc)
            sentence_counts.append(sc)
            char_counts.append(cc)
            mean_word_lengths.append(mwl)
            words_per_sentence.append(wps)

        dataframe[f"{self.output_prefix}_word_count"] = np.array(word_counts)
        dataframe[f"{self.output_prefix}_sentence_count"] = np.array(sentence_counts)
        dataframe[f"{self.output_prefix}_char_count"] = np.array(char_counts)
        dataframe[f"{self.output_prefix}_mean_word_length"] = np.array(mean_word_lengths)
        dataframe[f"{self.output_prefix}_words_per_sentence"] = np.array(words_per_sentence)

        # Corrected: FileStorage.write expects only the data argument
        storage.write(dataframe)
        self.logger.info(
            f"Statistics computation completed. Total records processed: {len(dataframe)}."
        )

        return [
            f"{self.output_prefix}_word_count",
            f"{self.output_prefix}_sentence_count",
            f"{self.output_prefix}_char_count",
            f"{self.output_prefix}_mean_word_length",
            f"{self.output_prefix}_words_per_sentence",
        ]


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
 
# 3. Instantiate operator
operator = TextDensityStats()

# 4. Run
operator.run(storage=storage.step())
