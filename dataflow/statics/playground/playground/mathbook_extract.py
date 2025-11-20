from dataflow.operators.knowledge_cleaning import MathBookQuestionExtract
from dataflow.serving import APIVLMServing_openai

class QuestionExtractPipeline:
    def __init__(self, 
                 llm_serving: APIVLMServing_openai,
                 api_url: str = "https://api.openai.com/v1", # end with /v1
                 key_name_of_api_key: str = "DF_API_KEY", # set in environment first: export DF_API_KEY="your_openai_api_key"
                 model_name: str = "o4-mini",
                 max_workers: int = 20
                 ):
        self.extractor = MathBookQuestionExtract(
            llm_serving=llm_serving,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            max_workers=max_workers
        )
        self.test_pdf = "../example_data/PDF2VQAPipeline/questionextract_test.pdf" 

    def forward(
        self,
        pdf_path: str,
        output_name: str,
        output_dir: str,
    ):
        self.extractor.run(
            input_pdf_file_path=pdf_path,
            output_file_name=output_name,
            output_folder=output_dir
        )

if __name__ == "__main__":
    llm_serving = APIVLMServing_openai(
        api_url="https://api.openai.com/v1",
        model_name="o4-mini",
        max_workers=20
    )

    pipeline = QuestionExtractPipeline(llm_serving=llm_serving)
    pipeline.forward(
        pdf_path=pipeline.test_pdf,
        output_name="test_question_extract",
        output_dir="./output"
    )
