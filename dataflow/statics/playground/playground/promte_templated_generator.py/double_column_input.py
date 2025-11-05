from dataflow.operators.core_text import PromptTemplatedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from dataflow.prompts.core_text import StrFormatPrompt


class DoubleColumnInputTestCase():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/double_column_input.jsonl",
            file_name_prefix="double_column_input",
            cache_path="./cache",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",   
            model_name="gpt-4o"
        )

        self.prompt_template = StrFormatPrompt(
            f_str_template="What does a {input_roll} like to {input_term}?"
        )
        self.operator = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=self.prompt_template
        )

    def forward(self):
        self.operator.run(
            storage=self.storage.step(),
            input_roll="roll",
            input_term="term",
            output_key="answer",
        )
if __name__ == "__main__":

    model = DoubleColumnInputTestCase()
    model.forward()