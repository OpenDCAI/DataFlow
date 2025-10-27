from dataflow.operators.vqa import VQAExtractPdf2Img
from dataflow.operators.vqa import VQAExtractDocLayoutMinerU
from dataflow.operators.vqa import VQAExtractPicExtractor
from dataflow.operators.vqa import VQAExtractQAPairExtractor
from dataflow.operators.vqa import VQAExtractTag2Img
from dataflow.serving import APIVLMServing_openai
import os
import json
import re

def merge_qa_pair(question_jsonl, answer_jsonl, output_jsonl):
    with open(question_jsonl, 'r', encoding='utf-8') as q_file, open(answer_jsonl, 'r', encoding='utf-8') as a_file, open(output_jsonl, 'w', encoding='utf-8') as out_file:
        chapter_id = 0
        chapter_title = ""
        label = 1000000
        questions = {}
        for line in q_file:
            data = json.loads(line)
            label_match = re.search(r'\d+', data["label"])
            if label_match:
                data["label"] = label_match.group()
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if int(data["label"]) < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    data["chapter_title"] = chapter_title
            label = int(data["label"])
            data["chapter_id"] = chapter_id
            questions[(data["chapter_id"], data['label'])] = data
        
        chapter_id = 0
        chapter_title = ""
        label = 1000000
        answers = {}
        for line in a_file:
            data = json.loads(line)
            label_match = re.search(r'\d+', data["label"])
            if label_match:
                data["label"] = label_match.group()
            if data["chapter_title"] == "":
                data["chapter_title"] = chapter_title
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if int(data["label"]) < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    data["chapter_title"] = chapter_title
            label = int(data["label"])
            data["chapter_id"] = chapter_id
            answers[(data["chapter_id"], data['label'])] = data
        
        question_cnt = len(questions)
        answer_cnt = len(answers)
        print(f"Total questions: {question_cnt}")
        
        for label in questions:
            if label in answers:
                qa_pair = {
                    "question_chapter_title": questions[label]["chapter_title"],
                    "answer_chapter_title": answers[label]["chapter_title"],
                    "label": label[1],
                    "question": questions[label]["question"],
                    "answer": answers[label]["answer"],
                    "solution": answers[label].get("solution", "")
                }
                out_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"Merged QA pairs: {len(questions.keys() & answers.keys())}")


class VQA_long_distance_extract:
    def __init__(self, input_pdf_paths_jsonl_file: str, output_prefix: str = "doclay"):
        self.input_pdf_paths_jsonl_file = input_pdf_paths_jsonl_file
        self.output_prefix = output_prefix
        self.pdf2img = VQAExtractPdf2Img()
        self.doc_item_layout = VQAExtractDocLayoutMinerU()
        self.llm_serving = APIVLMServing_openai(
            api_url = "http://api.openai.com/v1",
            model_name = "o4-mini",
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
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理question pdf
            question_output_dir = os.path.join(output_dir, "question")
            os.makedirs(question_output_dir, exist_ok=True)
            output_json_path, output_layout_path = self.doc_item_layout.run(None, question_pdf_path, question_output_dir)
            self.pdf2img.run(None, output_layout_path, os.path.join(question_output_dir, "pdf_images"))
            
            self.pic_extractor.run(None, os.path.join(question_output_dir, "pdf_images"), subject, os.path.join(question_output_dir, "vqa_extract"))
            self.qapair_extractor.run(None, os.path.join(question_output_dir, "vqa_extract/vqa_extract.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract.jsonl"))
            
            self.piclabeltranslator.run(None, output_json_path, os.path.join(question_output_dir, "pdf_images"), os.path.join(output_dir, "vqa_extract_question_cut_images"), 
                                        os.path.join(question_output_dir, "vqa_extract/qapair_extract.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract_cut.jsonl"), os.path.join(question_output_dir, "vqa_extract/qapair_extract_cut.md"))

            # 处理answer pdf
            answer_output_dir = os.path.join(output_dir, "answer")
            os.makedirs(answer_output_dir, exist_ok=True)
            output_json_path, output_layout_path = self.doc_item_layout.run(None, answer_pdf_path, answer_output_dir)
            self.pdf2img.run(None, output_layout_path, os.path.join(answer_output_dir, "pdf_images"))
            self.pic_extractor.run(None, os.path.join(answer_output_dir, "pdf_images"), subject, os.path.join(answer_output_dir, "vqa_extract"))
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
            with open(os.path.join(output_dir, "vqa_extract_qa_pair.jsonl"), "r", encoding="utf-8") as qa_file, open(os.path.join(output_dir, "vqa_extract_qa_pair.md"), "w", encoding="utf-8") as md_file:
                chapter_title = ""
                for line in qa_file:
                    data = json.loads(line)
                    if data['question_chapter_title'] != chapter_title:
                        md_file.write(f"### Chapter: {data['question_chapter_title']}\n\n")
                        chapter_title = data['question_chapter_title']
                    md_file.write(f"**Q{data['label']}**: {data['question']}\n\n")
                    md_file.write(f"**A{data['label']}**: {data['answer']}\n\n")
                    md_file.write(f"**Solution{data['label']}**: {data.get('solution', '')}\n\n")
                    md_file.write("---\n\n")


if __name__ == "__main__":
    vqa_extract = VQA_long_distance_extract("./dataflow/example/VQA/vqa_extract_long_distance_test.jsonl") # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    vqa_extract.run()