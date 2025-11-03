import json
from dataflow.serving import APILLMServing_request
from dataflow.operators.vqa import QAExtractor
import os
import re
from dataflow.operators.vqa import VQAExtractDocLayoutMinerU
from dataflow.utils.vqa.format_utils import merge_qa_pair, jsonl_to_md
from pathlib import Path
import shutil

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
        self.doc_item_layout = VQAExtractDocLayoutMinerU('vlm-vllm-engine')
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                key_name_of_api_key="DF_API_KEY",
                model_name="gemini-2.5-pro",
                max_workers=100,
            )
        self.qa_extractor = QAExtractor(llm_serving=self.llm_serving)
        
    def run(self):
        with open(self.input_jsonl_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        layout_entries = []  # [(mode, subject, json_path, output_dir, interleaved, output_root), ...]
        outputs = {}

        # === 阶段1：Layout 批量提取 ===
        print("=== [Stage 1] Extracting layouts for all PDFs ===")
        for data in lines:
            question_pdf_path = data["question_pdf_path"]
            answer_pdf_path = data["answer_pdf_path"]
            subject = data.get("subject", "General")
            output_root = data.get("output_dir", "../vqa_output")
            os.makedirs(output_root, exist_ok=True)
            interleaved = (question_pdf_path == answer_pdf_path)

            # Question
            q_outdir = os.path.join(output_root, "question")
            os.makedirs(q_outdir, exist_ok=True)
            q_json_path, _ = self.doc_item_layout.run(None, question_pdf_path, q_outdir)
            layout_entries.append(("question", subject, q_json_path, q_outdir, interleaved, output_root))

            if not interleaved:
                a_outdir = os.path.join(output_root, "answer")
                os.makedirs(a_outdir, exist_ok=True)
                a_json_path, _ = self.doc_item_layout.run(None, answer_pdf_path, a_outdir)
                layout_entries.append(("answer", subject, a_json_path, a_outdir, interleaved, output_root))

            outputs[output_root] = {
                "subject": subject,
                "interleaved": interleaved,
                "q_json_path": q_json_path,
                "a_json_path": None if interleaved else a_json_path,
            }

        print(f"✅ Layout extraction done. Total: {len(layout_entries)}")

        # === 阶段2：批量 QA 提取 ===
        print("=== [Stage 2] Batch QA extraction ===")
        input_json_paths = [e[2] for e in layout_entries]
        subjects = [e[1] for e in layout_entries]

        responses = self.qa_extractor.run(
            storage=None,
            input_json_paths=input_json_paths,
            input_subject = subjects[0]
        )

        assert len(responses) == len(layout_entries), "Response count mismatch!"

        # 写出 response.txt
        for (mode, subject, json_path, out_dir, interleaved, output_root), resp in zip(layout_entries, responses):
            response_path = os.path.join(out_dir, f"vqa_extracted_{mode}_response.txt")
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(resp)

        print(f"✅ QA batch extraction complete: {len(responses)} responses written.")

        # === 阶段3：后处理 ===
        print("=== [Stage 3] Converting and merging ===")
        for output_root, info in outputs.items():
            subject = info["subject"]
            interleaved = info["interleaved"]
            q_json_path = info["q_json_path"]
            a_json_path = info["a_json_path"]

            # === QUESTION ===
            q_response_path = os.path.join(output_root, "question/vqa_extracted_question_response.txt")
            with open(q_response_path, 'r', encoding='utf-8') as f:
                q_response = f.read()
            q_output_file = q_json_path.replace('.json', '_converted.json')
            q_list = convert_response(q_response, q_output_file, "question_images")
            src_dir = os.path.join(output_root, 'question', Path(q_json_path).stem).replace('_content_list','')
            src_images = os.path.join(src_dir, 'vlm', 'images')
            dst_images = os.path.join(output_root, 'question_images')
            try:
                if os.path.exists(src_images):
                    if os.path.exists(dst_images):
                        shutil.rmtree(dst_images)
                    shutil.copytree(src_images, dst_images)
                else:
                    print(f"Warning: source images dir does not exist: {src_images}")
            except Exception as e:
                print(f"Warning: failed to copy images from {src_images} to {dst_images}: {e}")

            q_jsonl_path = os.path.join(output_root, "vqa_extracted_questions.jsonl")
            with open(q_jsonl_path, 'w', encoding='utf-8') as f:
                for item in q_list:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            # === ANSWER ===
            if not interleaved and a_json_path:
                a_response_path = os.path.join(output_root, "answer/vqa_extracted_answer_response.txt")
                with open(a_response_path, 'r', encoding='utf-8') as f:
                    a_response = f.read()

                a_output_file = a_json_path.replace('.json', '_converted.json')
                a_list = convert_response(a_response, a_output_file, "answer_images")
                src_dir = os.path.join(output_root, 'answer', Path(a_json_path).stem).replace('_content_list','')
                src_images = os.path.join(src_dir, 'vlm', 'images')
                dst_images = os.path.join(output_root, 'answer_images')
                try:
                    if os.path.exists(src_images):
                        if os.path.exists(dst_images):
                            shutil.rmtree(dst_images)
                        shutil.copytree(src_images, dst_images)
                    else:
                        print(f"Warning: source images dir does not exist: {src_images}")
                except Exception as e:
                    print(f"Warning: failed to copy images from {src_images} to {dst_images}: {e}")

                a_jsonl_path = os.path.join(output_root, "vqa_extracted_answers.jsonl")
                with open(a_jsonl_path, 'w', encoding='utf-8') as f:
                    for item in a_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                merged_jsonl = os.path.join(output_root, "vqa_merged_qa_pairs.jsonl")
                merge_qa_pair(q_jsonl_path, a_jsonl_path, merged_jsonl)
            else:
                merged_jsonl = os.path.join(output_root, "vqa_merged_qa_pairs.jsonl")
                os.system(f"cp {q_jsonl_path} {merged_jsonl}")

            # 过滤jsonl，只保留有question，并且answer或solution不为空的条目
            filtered_items = []
            total_count = 0
            with open(merged_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    total_count += 1
                    item = json.loads(line)
                    if item.get('question','').strip() and (item.get('answer','').strip() or item.get('solution','').strip()):
                        filtered_items.append(item)

            print(f"Before filter: {total_count}")
            print(f"After filter: {len(filtered_items)}")

            with open(os.path.join(output_root, "vqa_filtered_qa_pairs.jsonl"), 'w', encoding='utf-8') as f:
                for item in filtered_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            md_output = os.path.join(output_root, "vqa_merged_qa_pairs.md")
            jsonl_to_md(os.path.join(output_root, "vqa_filtered_qa_pairs.jsonl"), md_output)
            print(f"✅ Completed: {output_root}")

        print("🎉 All PDFs processed successfully.")



if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    vqa_extractor = VQA_extract("/data1/djw/vqa_ode.jsonl")
    vqa_extractor.run()