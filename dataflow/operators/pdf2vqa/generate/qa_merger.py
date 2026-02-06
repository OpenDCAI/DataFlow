import os
import json
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.pdf2vqa.format_utils import merge_qa_pair, jsonl_to_md

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
                "并转换为Markdown格式。"
            )
        else:
            return (
                "QA pair merging operator."
                "Merges question and answer QA lists to generate final QA pair files,"
                "and converts them to Markdown format."
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

        storage.write(dataframe)