from dataflow.operators.filter import (
    AnswerNgramFilter
)

from dataflow.utils.storage import FileStorage
import pytest

class RemoteDataLoader():
    def __init__(self):
        
        self.storage_1 = FileStorage(
            first_entry_file_name="hf:openai/gsm8k:main:train",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step_hf",
            cache_type="jsonl",
        )

        self.storage_2 = FileStorage(
            first_entry_file_name="ms:modelscope/gsm8k:train",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step_ms",
            cache_type="jsonl",
        )

        self.answer_ngram_filter_step1 = AnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
    def forward(self):
        self.answer_ngram_filter_step1.run(
            storage = self.storage_1.step(),
            question_key = "question",
            answer_key = "answer"
        )

        self.answer_ngram_filter_step1.run(
            storage = self.storage_2.step(),
            question_key = "question",
            answer_key = "answer"
        )
@pytest.mark.gpu  
def test_remote_data_loader():
    """
    Test function to run the RemoteDataLoader
    """
    import pytest
    try:
        loader = RemoteDataLoader()
        loader.forward()
    except Exception as e:
        pytest.fail(f"RemoteDataLoader execution failed with error: {e}")

if __name__ == "__main__":
    loader = RemoteDataLoader()
    loader.forward()