from dataflow.operators.core_text import FormatStrPromptedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from dataflow.prompts.core_text import FormatStrPrompt


class DoubleColumnInputTestCase():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/double_column_input.json",
            file_name_prefix="double_column_input",
            cache_path="./cache",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",   
            model_name="gpt-4o"
        )

        self.prompt_template = FormatStrPrompt(
            f_str_template="What does a {input_role} like to {input_term}?"
        )
        self.operator = FormatStrPromptedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=self.prompt_template
        )

    def forward(self):
        self.operator.run(
            storage=self.storage.step(),
            input_role="role",
            input_term="term",
            output_key="answer",
        )
if __name__ == "__main__":

    model = DoubleColumnInputTestCase()
    model.forward()