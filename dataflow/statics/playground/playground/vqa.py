from dataflow.operators.generate.Vqa.PromptedVQAGenerator import PromptedVQAGenerator
from dataflow.serving.APIVLMServing_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage

class Vqa_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="dataflow/example/Vqa/pic_path.json",
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="json",
        )
        self.llm_serving = APIVLMServing_openai(
            model_name="o4-mini",
            api_url="https://api.openai.com/v1", # openai api url
            key_name_of_api_key="DF_API_KEY",
        )
        self.vqa_generate = PromptedVQAGenerator(self.llm_serving)

    def forward(self):
        self.vqa_generate.run(
            storage = self.storage.step(),
            input_key = "raw_content",
        )

if __name__ == "__main__":
    vqa_generator = Vqa_generator()
    vqa_generator.forward()