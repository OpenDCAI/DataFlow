import json
from dataflow.serving import APILLMServing_request
from dataflow.operators.vqa import QAExtractor
import os
import re
from dataflow.operators.vqa import VQAExtractDocLayoutMinerU
from dataflow.utils.vqa.format_utils import merge_qa_pair, jsonl_to_md
from pathlib import Path

def id_to_text(input_ids, input_json, image_prefix="images"):
    texts = []
    id_list = input_ids.replace(' ', '').split(',')
    for id in id_list:
        try: 
            int(id)
        except:
            continue
        if int(id) < len(input_json):
            try:
                item = input_json[int(id)]
            except:
                continue
            if 'text' in item:
                texts.append(item['text'])
            elif 'img_path' in item:
                try:
                    img_path = item.get('img_path', '')
                    img_name = os.path.basename(img_path)
                    new_path = f"{image_prefix}/{img_name}"
                    texts.append(f"![{' '.join(item.get('image_caption','image'))}]({new_path})")
                except:
                    pass
            elif item.get('type','') == 'list':
                if item['sub_type'] == 'text':
                    try:
                        texts.append(input_json[int(id)]['list_items'].pop(0))
                    except:
                        pass

    return '\n'.join(texts)

def convert_response(input_response, input_json_path, image_prefix="images"):
    qa_list = []
    with open(input_json_path, 'r') as infile:
        input_json = list(json.load(infile))
    # 提取title
    for chapter_block in re.findall(r'<chapter>(.*?)</chapter>', input_response, flags=re.DOTALL):
        title = re.search(r'<title>(.*?)</title>', chapter_block, flags=re.DOTALL)
        if title:
            chapter_title = id_to_text(title.group(1).strip(), input_json, image_prefix)
        else:
            chapter_title = ""
        # 找出所有 qa_pair 块
        for pair in re.findall(r'<qa_pair>(.*?)</qa_pair>', chapter_block, flags=re.DOTALL):
            # 提取 question 部分
            q_match = re.search(r'<question>(.*?)</question>', pair, flags=re.DOTALL)
            # 提取 answer 部分
            a_match = re.search(r'<answer>(.*?)</answer>', pair, flags=re.DOTALL)
            # 提取solution部分
            s_match = re.search(r'<solution>(.*?)</solution>', pair, flags=re.DOTALL)
            # 提取label
            label_match = re.search(r'<label>(.*?)</label>', pair, flags=re.DOTALL)
            if not ((q_match and label_match) or (a_match and label_match) or (s_match and label_match)):
                continue
            label = label_match.group(1).strip()
            qa_list.append({
                'question': id_to_text(q_match.group(1).strip(), input_json, image_prefix) if q_match else "",
                'answer': a_match.group(1).strip() if a_match else "",
                'solution': id_to_text(s_match.group(1).strip(), input_json, image_prefix) if s_match else "",
                'label': label,
                'chapter_title': chapter_title
            })
    return qa_list

class VQA_extract:
    def __init__(self, input_jsonl_file: str):
        self.input_jsonl_file = input_jsonl_file
        self.doc_item_layout = VQAExtractDocLayoutMinerU()
        self.llm_serving = APILLMServing_request(
                api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                key_name_of_api_key="DF_API_KEY",
                model_name="gemini-2.5-pro",
                max_workers=100,
            )
        self.qa_extractor = QAExtractor(llm_serving=self.llm_serving)
        
    def run(self):
        with open(self.input_jsonl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            question_pdf_path = data["question_pdf_path"]
            answer_pdf_path = data["answer_pdf_path"]
            subject = data.get("subject", "General")
            output_dir = data.get("output_dir", f"../vqa_output")
            os.makedirs(output_dir, exist_ok=True)
            
            interleaved = (question_pdf_path == answer_pdf_path)

            # === QUESTION ===
            print(f"Processing QUESTION PDF: {question_pdf_path}")
            question_output_dir = os.path.join(output_dir, "question")
            os.makedirs(question_output_dir, exist_ok=True)

            q_json_path, q_layout_path = self.doc_item_layout.run(None, question_pdf_path, question_output_dir)
            
            q_input_file = q_json_path
            q_output_file = q_json_path.replace('.json', '_converted.json')

            # Extract question QA
            q_response = self.qa_extractor.run(
                storage=None,
                input_json_path=q_input_file,
                input_subject=subject,
            )

            q_response_path = os.path.join(output_dir, "vqa_extracted_question_response.txt")
            with open(q_response_path, 'w', encoding='utf-8') as outfile:
                outfile.write(q_response)

            with open(q_response_path, 'r', encoding='utf-8') as infile:
                q_response = infile.read()
            q_list = convert_response(q_response, q_output_file, "question_images")

            os.system(f"cp -r {question_output_dir}/{Path(question_pdf_path).stem}/vlm/images {output_dir}/question_images")

            q_jsonl_path = os.path.join(output_dir, "vqa_extracted_questions.jsonl")
            with open(q_jsonl_path, 'w', encoding='utf-8') as outfile:
                for item in q_list:
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"Question extraction completed. Total questions extracted: {len(q_list)}")

            # === ANSWER ===
            if not interleaved:
                print(f"Processing ANSWER PDF: {answer_pdf_path}")
                answer_output_dir = os.path.join(output_dir, "answer")
                os.makedirs(answer_output_dir, exist_ok=True)

                a_json_path, a_layout_path = self.doc_item_layout.run(None, answer_pdf_path, answer_output_dir)

                a_input_file = a_json_path
                a_output_file = a_json_path.replace('.json', '_converted.json')

                a_response = self.qa_extractor.run(
                    storage=None,
                    input_json_path=a_input_file,
                    input_subject=subject,
                )

                a_response_path = os.path.join(output_dir, "vqa_extracted_answer_response.txt")
                with open(a_response_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(a_response)

                with open(a_response_path, 'r', encoding='utf-8') as infile:
                    a_response = infile.read()
                a_list = convert_response(a_response, a_output_file, "answer_images")

                os.system(f"cp -r {answer_output_dir}/{Path(answer_pdf_path).stem}/vlm/images {output_dir}/answer_images")

                a_jsonl_path = os.path.join(output_dir, "vqa_extracted_answers.jsonl")
                with open(a_jsonl_path, 'w', encoding='utf-8') as outfile:
                    for item in a_list:
                        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

                print(f"Answer extraction completed. Total answers extracted: {len(a_list)}")

                # === MERGE Q&A ===
                merged_jsonl = os.path.join(output_dir, "vqa_merged_qa_pairs.jsonl")
                merge_qa_pair(q_jsonl_path, a_jsonl_path, merged_jsonl)
            else:
                # 如果是interleaved模式，直接将question的jsonl作为merged jsonl
                merged_jsonl = os.path.join(output_dir, "vqa_merged_qa_pairs.jsonl")
                os.system(f"cp {q_jsonl_path} {merged_jsonl}")

            # === EXPORT TO MARKDOWN ===
            md_output = os.path.join(output_dir, "vqa_merged_qa_pairs.md")
            jsonl_to_md(merged_jsonl, md_output)

            print(f"✅ Completed: {output_dir}")


if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    vqa_extractor = VQA_extract("../example_data/VQA/vqa_extract_long_distance_test.jsonl")
    vqa_extractor.run()