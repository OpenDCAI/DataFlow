from {{cookiecutter.package_name}}.operators.core.my_format_str_prompted_generator import MyFormatStrPromptedGenerator
from {{cookiecutter.package_name}}.operators.core.my_prompted_generator import MyPromptedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.pipeline import PipelineABC

from {{cookiecutter.package_name}}.prompts.core import MyFormatStringPrompt

class MySimplePipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name="../data/example_data.json",
            file_name_prefix="dataflow{{cookiecutter.package_name}}",
            cache_path="./cache",
            cache_type="jsonl",
        )
        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",   
            model_name="gpt-4o"
        )
        self.prompt_template = MyFormatStringPrompt(
            f_str_template="What does a {input_roll} like to {input_term}?"
        )
        self.op1 = MyFormatStrPromptedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=self.prompt_template
        )
        self.op2 = MyPromptedGenerator(
            llm_serving = self.llm_serving,
            system_prompt="Summary this answer"
        )

    def forward(self):
        self.op1.run(
            storage=self.storage.step(),
            input_roll="roll",
            input_term="term",
            output_key="answer",
        )
        self.op2.run(
            storage=self.storage.step(),
            input_key="answer",
            output_key="summary_answer"
        )

if __name__ == "__main__":
    model = MySimplePipeline()
    model.compile()
    model.forward()