from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
import pandas as pd
import re

@OPERATOR_REGISTRY.register()
class MathVQAExtractQAPairExtractor(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    
    def extract_qa_pairs_from_text(self, page: int, text: str):
        """
        从一段 response 文本中提取所有 <qa_pair>…</qa_pair>
        并返回 [{'question': ..., 'answer': ...}, …]
        """
        qa_list = []
        # 找出所有 qa_pair 块
        for pair in re.findall(r'<qa_pair>(.*?)</qa_pair>', text, flags=re.DOTALL):
            # 提取 question 部分
            q_match = re.search(r'<question>(.*?)</question>', pair, flags=re.DOTALL)
            # 提取 answer 部分
            a_match = re.search(r'<answer>(.*?)</answer>', pair, flags=re.DOTALL)
            if not (q_match and a_match):
                continue
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            qa_list.append({
                'page': page,
                'question': question,
                'answer': answer
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