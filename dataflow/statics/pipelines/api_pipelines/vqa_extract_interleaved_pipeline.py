from dataflow.operators.vqa import VQAExtractPdf2Img
from dataflow.operators.vqa import VQAExtractDocLayoutMinerU
from dataflow.operators.vqa import VQAExtractPicExtractor
from dataflow.operators.vqa import VQAExtractQAPairExtractor
from dataflow.operators.vqa import VQAExtractTag2Img
from dataflow.operators.vqa import VQAClipHeader
from dataflow.operators.vqa import VQAConcatenateImages
from dataflow.serving import APIVLMServing_openai
from dataflow.serving import APILLMServing_request
import os
import json

from dataflow.utils.storage import FileStorage
from dataflow.operators.general_text.filter.minhash_deduplicate_filter import MinHashDeduplicateFilter


class VQA_interleaved_extract:
    def __init__(self, input_pdf_paths_jsonl_file: str, output_prefix: str = "doclay"):
        # self.pdf_path = pdf_path
        # self.subject = subject
        self.input_pdf_paths_jsonl_file = input_pdf_paths_jsonl_file
        self.output_prefix = output_prefix
        self.pdf2img = VQAExtractPdf2Img()
        self.doc_item_layout = VQAExtractDocLayoutMinerU()
        self.clip_header = VQAClipHeader()
        self.concatenate_images = VQAConcatenateImages()
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
        self.qapair_extractor = VQAExtractQAPairExtractor()
        
    def run(self):
        with open(self.input_pdf_paths_jsonl_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            pdf_path = data["pdf_path"]
            subject = data.get("subject", "")
            output_dir = data.get("output_dir", "../vqa")

            os.makedirs(output_dir, exist_ok=True)
            output_json_path, output_layout_path = self.doc_item_layout.run(None, pdf_path, output_dir)
            self.pdf2img.run(None, output_layout_path, os.path.join(output_dir, "pdf_images"))
            self.clip_header.run(None, os.path.join(output_dir, "pdf_images"), output_json_path, os.path.join(output_dir, "cropped_images"))
            self.concatenate_images.run(None, os.path.join(output_dir, "cropped_images"), os.path.join(output_dir, "concatenated_images"))
            
            pic_extractor = VQAExtractPicExtractor(self.llm_serving, subject=subject)
            pic_extractor.run(None, os.path.join(output_dir, "concatenated_images"), os.path.join(output_dir, "vqa_extract"))
            self.qapair_extractor.run(None, os.path.join(output_dir, "vqa_extract/vqa_extract.jsonl"), os.path.join(output_dir, "vqa_extract/qapair_extract.jsonl"))
            
            piclabeltranslator = VQAExtractTag2Img(output_json_path, os.path.join(output_dir, "pdf_images"), os.path.join(output_dir, "vqa_extract_cut_images"), layout_prefix="doclay_concatenated_", image_prefix='page_')
            piclabeltranslator.run(None, os.path.join(output_dir, "vqa_extract/qapair_extract.jsonl"), os.path.join(output_dir, "vqa_extract/qapair_extract_cut.jsonl"), os.path.join(output_dir, "vqa_extract/qapair_extract_cut.md"))

            # 将jsonl文件先倒转过来 (因为有时候最后的答案是完整的，而前面的不完整)
            with open(os.path.join(output_dir, "vqa_extract/qapair_extract_cut.jsonl"), "r") as f:
                lines = f.readlines()
            lines.reverse()
            with open(os.path.join(output_dir, "vqa_extract/qapair_extract_cut.jsonl"), "w") as f:
                f.writelines(lines)
            # 去重
            vqa_deduplicate = VQA_deduplicate(os.path.join(output_dir, "vqa_extract/qapair_extract_cut.jsonl"), cache_path=output_dir)
            vqa_deduplicate.run()
            # 把结果再倒转回来
            with open(os.path.join(output_dir, "vqa_step1.jsonl"), "r") as f:
                lines = f.readlines()
            lines.reverse()
            with open(os.path.join(output_dir, "vqa_final_result.jsonl"), "w") as f:
                f.writelines(lines)
            os.remove(os.path.join(output_dir, "vqa_step1.jsonl"))

class VQA_deduplicate:
    def __init__(self, input_path: str, cache_path: str = "../vqa"):
        self.storage = FileStorage(
            first_entry_file_name=input_path,
            cache_path=cache_path,
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        self.deduplicate = MinHashDeduplicateFilter(num_perm=64, threshold=0.6, ngram=3) # 参数可以自己调
        
    def run(self):
        self.deduplicate.run(self.storage.step(), input_keys=["question", "answer"])

if __name__ == "__main__":
    vqa_extract = VQA_interleaved_extract("./dataflow/example/VQA/vqa_extract_test.jsonl") # jsonl中每一行包含pdf_path, subject (math, physics, chemistry, ...), output_dir
    vqa_extract.run()