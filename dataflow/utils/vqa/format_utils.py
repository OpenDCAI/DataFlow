import json
import re
import regex

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
            
            try:
                data["label"] = int(data["label"])
            except:
                continue
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_id"] = chapter_id
            # 删除title中的空格，标点符号（包括中文和英文）
            data["chapter_title"] = regex.sub(r'[\p{P}\s]+', '', data["chapter_title"])
            if data['label'] > 0:
                questions[(data["chapter_title"], data['label'])] = data
        
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
                
            try:
                data["label"] = int(data["label"])
            except:
                continue
            
            if data["chapter_title"] != "" and data["chapter_title"] != chapter_title:
                if data["label"] < label:
                    chapter_id += 1
                    chapter_title = data["chapter_title"]
                else:
                    # 如果题号增加，章节标题却发生变化，说明可能错误提取了子标题。因此继续使用之前的章节标题。
                    data["chapter_title"] = chapter_title
            label = data["label"]
            data["chapter_id"] = chapter_id
            # 删除title中的空格，标点符号（包括中文和英文）
            data["chapter_title"] = regex.sub(r'[\p{P}\s]+', '', data["chapter_title"])
            # 动态更新，防止错误的重复label覆盖掉之前的solution或answer
            if data['label'] > 0:
                if not answers.get((data["chapter_title"], data['label'])):
                    answers[(data["chapter_title"], data['label'])] = data
                else:
                    if not answers[(data["chapter_title"], data['label'])].get("solution") and data.get("solution"):
                        answers[(data["chapter_title"], data['label'])]["solution"] = data["solution"]
                    if not answers[(data["chapter_title"], data['label'])].get("answer") and data.get("answer"):
                        answers[(data["chapter_title"], data['label'])]["answer"] = data["answer"]
      
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
        
def jsonl_to_md(jsonl_file, md_file):
    with open(jsonl_file, 'r', encoding='utf-8') as in_file, open(md_file, 'w', encoding='utf-8') as out_file:
        for line in in_file:
            data = json.loads(line)
            out_file.write(f"### Question {data['label']}\n\n")
            out_file.write(f"{data['question']}\n\n")
            out_file.write(f"**Answer:** {data['answer']}\n\n")
            if data.get('solution'):
                out_file.write(f"**Solution:**\n\n{data['solution']}\n\n")
            out_file.write("---\n\n")