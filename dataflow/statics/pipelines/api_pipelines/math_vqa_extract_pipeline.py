from dataflow.operators.knowledge_cleaning import MathVQAExtractPdf2Img
from dataflow.operators.knowledge_cleaning import MathVQAExtractDocLayout
from dataflow.operators.knowledge_cleaning import MathVQAExtractPicExtractor
from dataflow.operators.knowledge_cleaning import MathVQAExtractQAPairExtractor
from dataflow.operators.knowledge_cleaning import MathVQAExtractTag2Img
from dataflow.serving import APIVLMServing_openai
from dataflow.serving import APILLMServing_request
import os


class VQA_extract:
    def __init__(self, pdf_path: str, output_prefix: str = "doclay"):
        self.pdf_path = pdf_path
        self.output_prefix = output_prefix
        self.pdf2img = MathVQAExtractPdf2Img()
        self.doc_item_layout = MathVQAExtractDocLayout(os.path.join(self.pdf_path, "./model/doclayout_yolo_docstructbench_imgsz1024.pt"))
        self.llm_serving = APIVLMServing_openai(
            api_url = "https://api.openai.com/v1",
            model_name = "o4-mini",
            max_workers = 100,
        )
        self.text_serving = APILLMServing_request(
            api_url = "https://api.openai.com/v1/chat/completions",
            model_name = "o4-mini",
            max_workers = 10,
        )
        self.pic_extractor = MathVQAExtractPicExtractor(self.llm_serving)
        self.qapair_extractor = MathVQAExtractQAPairExtractor()
        self.piclabeltranslator = MathVQAExtractTag2Img("./layout_images/json", "./pdf_images", "./vqa_extract_cut_images")
    def run(self):
        self.pdf2img.run(self.pdf_path, "./pdf_images")
        self.doc_item_layout.run("./pdf_images", "./layout_images", self.output_prefix)
        self.pic_extractor.run("./layout_images", "./vqa_extract")
        self.qapair_extractor.run("./vqa_extract/vqa_extract.jsonl", "./vqa_extract/qapair_extract.jsonl")
        self.piclabeltranslator.run("./vqa_extract/qapair_extract.jsonl", "./vqa_extract/qapair_extract_cut.jsonl", "./vqa_extract/qapair_extract_cut.md")



if __name__ == "__main__":
    vqa_extract = VQA_extract("./dataflow/example/KBCleaningPipeline/questionextract_test.pdf")
    vqa_extract.run()