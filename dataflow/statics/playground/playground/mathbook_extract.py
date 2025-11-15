from dataflow.operators.knowledge_cleaning.generate.mathbook_question_extract import MathBookQuestionExtract
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai

class QuestionExtractPipeline:
    def __init__(self, llm_serving: APIVLMServing_openai):
        self.extractor = MathBookQuestionExtract(llm_serving)
        self.test_pdf = "../example_data/KBCleaningPipeline/questionextract_test.pdf" 

    def forward(
        self,
        pdf_path: str,
        output_name: str,
        output_dir: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "o4-mini",
        max_workers: int = 20
    ):
        self.extractor.run(
            pdf_file_path=pdf_path,
            output_file_name=output_name,
            output_folder=output_dir,
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            max_workers=max_workers
        )

if __name__ == "__main__":
    # 1. initialize LLM Serving
    llm_serving = APIVLMServing_openai(
        api_url="https://api.openai.com/v1",  # end with /v1, DO NOT add /chat/completions
        model_name="o4-mini",      # recommend using strong reasoning model
        max_workers=20             # number of concurrent requests
    )

    # 2. construct and run pipeline
    pipeline = QuestionExtractPipeline(llm_serving)
    pipeline.forward(
        pdf_path=pipeline.test_pdf,
        output_name="test_question_extract",
        output_dir="./output"
    )
