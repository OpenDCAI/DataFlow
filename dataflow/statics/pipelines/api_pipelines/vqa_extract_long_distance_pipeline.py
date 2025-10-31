from dataflow.operators.vqa import VQAExtractPdf2Img
from dataflow.operators.vqa import VQAExtractDocLayoutMinerU
from dataflow.operators.vqa import VQAExtractPicExtractor
from dataflow.operators.vqa import VQAExtractQAPairExtractor
from dataflow.operators.vqa import VQAExtractTag2Img
from dataflow.serving import APIVLMServing_openai
from dataflow.utils.vqa.format_utils import merge_qa_pair, jsonl_to_md
import os
import json
import re
import regex


class VQA_long_distance_extract:
    def __init__(self, input_pdf_paths_jsonl_file: str):
        self.input_pdf_paths_jsonl_file = input_pdf_paths_jsonl_file
        self.pdf2img = VQAExtractPdf2Img()
        self.doc_item_layout = VQAExtractDocLayoutMinerU()
        self.llm_serving = APIVLMServing_openai(
            api_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
            key_name_of_api_key="DF_API_KEY",
            model_name = "gemini-2.5-pro",
            max_workers = 100,
        )
        self.pic_extractor = VQAExtractPicExtractor(self.llm_serving, interleaved=False)
        self.qapair_extractor = VQAExtractQAPairExtractor()
        self.piclabeltranslator = VQAExtractTag2Img(layout_prefix="doclay_concatenated_", image_prefix='page_')
        
    def run(self):
        with open(self.input_pdf_paths_jsonl_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            question_pdf_path = data["question_pdf_path"]
            answer_pdf_path = data["answer_pdf_path"]
            subject = data.get("subject", "")
            output_dir = data.get("output_dir", "../vqa")
            example_title = data.get("example_title", "")
            os.makedirs(output_dir, exist_ok=True)
            
            # 首先确保question_pdf_path和answer_pdf_path存在
            if not os.path.exists(question_pdf_path):
                print(f"Question PDF path does not exist: {question_pdf_path}")
                continue
            if not os.path.exists(answer_pdf_path):
                print(f"Answer PDF path does not exist: {answer_pdf_path}")
                continue
            
            # 处理question pdf
            question_output_dir = os.path.join(output_dir, "question")
            os.makedirs(question_output_dir, exist_ok=True)
            output_json_path, output_layout_path = self.doc_item_layout.run(None, question_pdf_path, question_output_dir)
            self.pdf2img.run(None, output_layout_path, os.path.join(question_output_dir, "pdf_images"))
            
            self.pic_extractor.run(None, os.path.join(question_output_dir, "pdf_images"), subject, example_title, os.path.join(question_output_dir, "vqa_extract"))
            self.qapair_extractor.run(None, os.path.join(question_output_dir, "vqa_extract/vqa_extract.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract.jsonl"))
            
            self.piclabeltranslator.run(None, output_json_path, os.path.join(question_output_dir, "pdf_images"), os.path.join(output_dir, "vqa_extract_question_cut_images"), 
                                        os.path.join(question_output_dir, "vqa_extract/qapair_extract.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract_cut.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract_cut.md"))

            # 处理answer pdf
            answer_output_dir = os.path.join(output_dir, "answer")
            os.makedirs(answer_output_dir, exist_ok=True)
            output_json_path, output_layout_path = self.doc_item_layout.run(None, answer_pdf_path, answer_output_dir)
            self.pdf2img.run(None, output_layout_path, os.path.join(answer_output_dir, "pdf_images"))
            self.pic_extractor.run(None, os.path.join(answer_output_dir, "pdf_images"), subject, example_title, os.path.join(answer_output_dir, "vqa_extract"))
            self.qapair_extractor.run(None, os.path.join(answer_output_dir, "vqa_extract/vqa_extract.jsonl"), os.path.join(answer_output_dir, "vqa_extract/qapair_extract.jsonl"))
            self.piclabeltranslator.run(None, output_json_path, os.path.join(answer_output_dir, "pdf_images"), os.path.join(output_dir, "vqa_extract_answer_cut_images"), 
                                   os.path.join(answer_output_dir, "vqa_extract/qapair_extract.jsonl"), os.path.join(answer_output_dir, "vqa_extract/qapair_extract_cut.jsonl"), os.path.join(answer_output_dir, "vqa_extract/qapair_extract_cut.md"))
            
            # 合并question和answer的qapair_extract_cut.jsonl
            merge_qa_pair(
                os.path.join(question_output_dir, "vqa_extract/qapair_extract_cut.jsonl"),
                os.path.join(answer_output_dir, "vqa_extract/qapair_extract_cut.jsonl"),
                os.path.join(output_dir, "vqa_extract_qa_pair.jsonl")
            )
            
            # dump 成 markdown
            jsonl_to_md(
                os.path.join(output_dir, "vqa_extract_qa_pair.jsonl"),
                os.path.join(output_dir, "vqa_extract_qa_pair.md")
            )


if __name__ == "__main__":
    # 在 https://huggingface.co/datasets/OpenDCAI/dataflow-demo-VQA 中有完整示例数据
    vqa_extract = VQA_long_distance_extract("../example_data/VQA/vqa_extract_long_distance_test.jsonl") # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    vqa_extract.run()