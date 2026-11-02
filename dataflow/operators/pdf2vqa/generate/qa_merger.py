import os
import json
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.pdf2vqa.format_utils import merge_qa_pair, jsonl_to_md

import re

@OPERATOR_REGISTRY.register()
class QA_Merger(OperatorABC):
    def __init__(self, output_dir, strict_title_match=False):
        self.output_dir = output_dir
        self.strict_title_match = strict_title_match
        
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == 'zh':
            return (
                "QA对合并算子。"
                "将问题和答案的QA列表进行合并，生成最终的QA对文件，"
                "并转换为Markdown格式。\n"
                "输出参数：\n"
                "- output_merged_qalist_path_key：写入合并后QA对jsonl文件路径（output_dir/<input_name_key值>/merged_qa_pairs.jsonl）的字段名，无默认值\n"
                "- output_merged_md_path_key：写入合并后Markdown文件路径（output_dir/<input_name_key值>/merged_qa_pairs.md）的字段名，无默认值\n"
                "- output_qa_item_key：写入单个QA对（dict）的字段名，默认为'qa_item'；其中question/answer/solution内的图片路径被加上<input_name_key值>/前缀\n"
                "- 按output_qa_item_key展开：每个输入行变为每个QA对一行，其余字段被复制；无QA对的输入行保留一行且该字段为空\n"
                "- 返回None"
            )
        else:
            return (
                "QA pair merging operator."
                "Merges question and answer QA lists to generate final QA pair files,"
                "and converts them to Markdown format.\n"
                "Output Parameters:\n"
                "- output_merged_qalist_path_key: field that receives the merged QA-pair jsonl path (output_dir/<input_name_key value>/merged_qa_pairs.jsonl), no default\n"
                "- output_merged_md_path_key: field that receives the merged Markdown path (output_dir/<input_name_key value>/merged_qa_pairs.md), no default\n"
                "- output_qa_item_key: field that receives one QA pair (dict), default 'qa_item'; image paths inside its question/answer/solution are prefixed with <input_name_key value>/\n"
                "- Exploded on output_qa_item_key: each input row becomes one row per QA pair with the other fields copied; an input row with no QA pairs keeps one row with this field empty\n"
                "- Returns None"
            )
    
    def run(self, storage: DataFlowStorage,
            input_qalist_path_key,
            input_name_key,
            output_merged_qalist_path_key,
            output_merged_md_path_key,
            output_qa_item_key="qa_item"  # 新增：展开后的 QA 内容列名
            ):
        dataframe = storage.read("dataframe")
        
        # 为了能存储 list 对象，先初始化该列为 object 类型
        dataframe[output_qa_item_key] = None
        dataframe[output_qa_item_key] = dataframe[output_qa_item_key].astype(object)

        for idx, row in dataframe.iterrows():
            qa_list_path = row[input_qalist_path_key]
            name = row[input_name_key]
            
            output_merged_qalist_path = os.path.join(self.output_dir, name, "merged_qa_pairs.jsonl")
            merge_qa_pair(qa_list_path, output_merged_qalist_path, strict_title_match=self.strict_title_match)
            
            output_merged_md_path = os.path.join(self.output_dir, name, "merged_qa_pairs.md")
            jsonl_to_md(output_merged_qalist_path, output_merged_md_path)
            
            qa_pairs = []
            if os.path.exists(output_merged_qalist_path):
                with open(output_merged_qalist_path, 'r', encoding='utf-8') as f:
                    qa_pairs = [json.loads(line) for line in f]
            
            dataframe.at[idx, output_qa_item_key] = qa_pairs

            dataframe.loc[idx, output_merged_qalist_path_key] = output_merged_qalist_path
            dataframe.loc[idx, output_merged_md_path_key] = output_merged_md_path
            
        dataframe = dataframe.explode(output_qa_item_key).reset_index(drop=True)

        # 汇总jsonl中的图片路径需要将 ![alt](path) 中的 path 替换为 name/path
        def fix_image_paths(row):
            qa_item = row[output_qa_item_key]
            name_val = str(row[input_name_key])
            
            if isinstance(qa_item, dict):
                keys_to_check = ["question", "answer", "solution"]
                for key in keys_to_check:
                    if key in qa_item and isinstance(qa_item[key], str):
                        qa_item[key] = re.sub(
                            r'!\[(.*?)\]\((.*?)\)',
                            lambda m: f"![{m.group(1)}]({os.path.join(name_val, m.group(2))})",
                            qa_item[key]
                        )
            return qa_item

        dataframe[output_qa_item_key] = dataframe.apply(fix_image_paths, axis=1)

        storage.write(dataframe)