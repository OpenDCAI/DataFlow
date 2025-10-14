from dataflow.operators.knowledge_cleaning import MathVQAExtractPdf2Img
from dataflow.operators.knowledge_cleaning import MathVQAExtractDocLayout
from dataflow.operators.knowledge_cleaning import MathVQAExtractPicExtractor
from dataflow.operators.knowledge_cleaning import MathVQAExtractQAPairExtractor
from dataflow.operators.knowledge_cleaning import MathVQAExtractTag2Img
from dataflow.operators.knowledge_cleaning import MathVQAClipHeader
from dataflow.operators.knowledge_cleaning import MathVQAConcatenateImages
from dataflow.serving import APIVLMServing_openai
from dataflow.serving import APILLMServing_request
import os

from dataflow.utils.storage import FileStorage
from dataflow.operators.general_text.filter.minhash_deduplicate_filter import MinHashDeduplicateFilter


class VQA_extract:
    def __init__(self, pdf_path: str, subject:str, output_prefix: str = "doclay"):
        self.pdf_path = pdf_path
        self.subject = subject
        self.output_prefix = output_prefix
        self.pdf2img = MathVQAExtractPdf2Img()
        self.doc_item_layout = MathVQAExtractDocLayout("/mnt/DataFlow/djw/doclayout_yolo_docstructbench_imgsz1024.pt")
        self.clip_header = MathVQAClipHeader()
        self.concatenate_images = MathVQAConcatenateImages()
        self.llm_serving = APIVLMServing_openai(
            api_url = "http://123.129.219.111:3000/v1",
            model_name = "gpt-4o-mini",
            max_workers = 100,
        )
        # self.text_serving = APILLMServing_request(
        #     api_url = "http://123.129.219.111:3000/v1/chat/completions",
        #     model_name = "gpt-4o-mini",
        #     max_workers = 10,
        # )
        self.pic_extractor = MathVQAExtractPicExtractor(self.llm_serving)
        self.qapair_extractor = MathVQAExtractQAPairExtractor()
        self.piclabeltranslator = MathVQAExtractTag2Img("../vqa/layout_concatenated_images/json", "../vqa/concatenated_images", "../vqa/vqa_extract_cut_images", layout_prefix="doclay_concatenated_", image_prefix='concatenated_')
    def run(self):
        os.makedirs("../vqa", exist_ok=True)
        self.pdf2img.run(self.pdf_path, "../vqa/pdf_images")
        self.doc_item_layout.run("../vqa/pdf_images", "../vqa/layout_images", self.output_prefix)
        self.clip_header.run("../vqa/pdf_images", "../vqa/layout_images/json", "../vqa/cropped_images")
        self.concatenate_images.run("../vqa/cropped_images", "../vqa/concatenated_images")
        self.doc_item_layout.run("../vqa/concatenated_images", "../vqa/layout_concatenated_images", self.output_prefix)
        self.pic_extractor.run("../vqa/layout_concatenated_images", "../vqa/vqa_extract", subject=self.subject)
        self.qapair_extractor.run("../vqa/vqa_extract/vqa_extract.jsonl", "../vqa/vqa_extract/qapair_extract.jsonl")
        self.piclabeltranslator.run("../vqa/vqa_extract/qapair_extract.jsonl", "../vqa/vqa_extract/qapair_extract_cut.jsonl", "../vqa/vqa_extract/qapair_extract_cut.md")

class VQA_deduplicate:
    def __init__(self, input_path: str):
        self.storage = FileStorage(
            first_entry_file_name=input_path,
            cache_path="../vqa",
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        self.deduplicate = MinHashDeduplicateFilter(num_perm=64, threshold=0.6, ngram=3)
        
    def run(self):
        self.deduplicate.run(self.storage.step(), input_keys=["question", "answer"])

if __name__ == "__main__":
    vqa_extract = VQA_extract("./dataflow/example/KBCleaningPipeline/chemistry_test.pdf", subject="chemistry")
    vqa_extract.run()
    
    # 将jsonl文件先倒转过来 (因为有时候最后的答案是完整的，而前面的不完整)
    with open("../vqa/vqa_extract/qapair_extract_cut.jsonl", "r") as f:
        lines = f.readlines()
    lines.reverse()
    with open("../vqa/vqa_extract/qapair_extract_cut.jsonl", "w") as f:
        f.writelines(lines)
    # 去重
    vqa_deduplicate = VQA_deduplicate("../vqa/vqa_extract/qapair_extract_cut.jsonl")
    vqa_deduplicate.run()
    # 把结果再倒转回来
    with open("../vqa/vqa_step1.jsonl", "r") as f:
        lines = f.readlines()
    lines.reverse()
    with open("../vqa/vqa_final_result.jsonl", "w") as f:
        f.writelines(lines)
    os.remove("../vqa/vqa_step1.jsonl")