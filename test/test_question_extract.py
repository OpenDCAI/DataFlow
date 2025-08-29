from dataflow.operators.reasoning import *

class QuestionExtractPipeline():
    def __init__(self, llm_serving=None):
        self.mathbook_question_extract = MathBookQuestionExtract()
        self.test_pdf_file_path = "./dataflow/example/KBCleaningPipeline/questionextract_test.pdf"

    def forward(self, pdf_file_path: str, output_file_name: str, output_folder: str, api_url: str = "http://123.129.219.111:3000/v1", key_name_of_api_key: str = "DF_API_KEY", model_name: str = "o4-mini", max_workers: int = 500):
        self.mathbook_question_extract.run(pdf_file_path = pdf_file_path,
                                           output_file_name = output_file_name,
                                           output_folder = output_folder,
                                           api_url = api_url,
                                           key_name_of_api_key = key_name_of_api_key,
                                           model_name = model_name,
                                           max_workers = max_workers)


if __name__ == "__main__":
    pipeline = QuestionExtractPipeline()
    pipeline.forward(pdf_file_path = pipeline.test_pdf_file_path,
                     output_file_name = "test_question_extract",
                     output_folder = "./cache_local")