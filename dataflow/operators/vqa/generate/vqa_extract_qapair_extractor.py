from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
import pandas as pd
import re

@OPERATOR_REGISTRY.register()
class VQAExtractQAPairExtractor(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    def extract_qa_pairs_from_text(self, page: int, text: str):
        """
        从一段 response 文本中提取所有 <qa_pair>…</qa_pair>
        格式为：<chapter><title>MAIN_TITLE</title>
        <qa_pair><label>…</label><question>QUESTION_TEXT<pic>…</pic>…</question>
        <answer>ANSWER_TEXT<pic>…</pic>…</answer></qa_pair>
        <qa_pair><label>…</label><question>QUESTION_TEXT<pic>…</pic>…</question>
        <answer>ANSWER_TEXT<pic>…</pic>…</answer></qa_pair>
        </chapter>
        并返回 [{'question': ..., 'answer': ...}, …]
        """
        qa_list = []
        # 提取title
        for chapter_block in re.findall(r'<chapter>(.*?)</chapter>', text, flags=re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', chapter_block, flags=re.DOTALL)
            if title:
                chapter_title = title.group(1).strip()
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
                    'page': page,
                    'question': q_match.group(1).strip() if q_match else "",
                    'answer': a_match.group(1).strip() if a_match else "",
                    'solution': s_match.group(1).strip() if s_match else "",
                    'label': label,
                    'chapter_title': chapter_title
                })
        return qa_list
    
    def run(self, storage, input_vqa_extract_path: str, output_qa_path: str):

        # 从vqa_extract_path中读取jsonl文件
        df = pd.read_json(input_vqa_extract_path, lines=True)

        items = df[['page', 'response']].to_dict(orient='records')
        responses = [item['response'] for item in items]
        pages = [item['page'] for item in items]

        qa_pairs = []
        for page, response in zip(pages, responses):
            qa_pairs.extend(self.extract_qa_pairs_from_text(page, response))

        # 将qa_pairs保存为jsonl文件
        df = pd.DataFrame(qa_pairs)
        df.to_json(output_qa_path, orient="records", lines=True, force_ascii=False)
        return df