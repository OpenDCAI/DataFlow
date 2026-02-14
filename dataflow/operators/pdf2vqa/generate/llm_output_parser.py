import os
import json
import re
import shutil
from pathlib import Path
from typing import Literal
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class LLMOutputParser(OperatorABC):
    def __init__(self, 
                 output_dir,
                 intermediate_dir: str = "intermediate",
                 ):
        self.logger = get_logger()
        self.output_dir = output_dir
        self.intermediate_dir = intermediate_dir
        
    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == 'zh':
            return (
                "LLM输出解析算子。"
                "将LLM生成的包含题目和答案ID的响应文本，"
                "转换为结构化的QA列表，并复制相关图片到输出目录。"
            )
        else:
            return (
                "LLM output parsing operator."
                "Converts LLM-generated response text containing question and answer IDs"
                "into a structured QA list and copies related images to the output directory."
            )
    
    def _id_to_text(self, input_ids, input_json, image_prefix="images"):
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
    
    def _convert_response(self, input_response, input_json_path, image_prefix="images"):
        qa_list = []
        with open(input_json_path, 'r') as infile:
            input_json = list(json.load(infile))
        # 提取title
        for chapter_block in re.findall(r'<chapter>(.*?)</chapter>', input_response, flags=re.DOTALL):
            title = re.search(r'<title>(.*?)</title>', chapter_block, flags=re.DOTALL)
            if title:
                chapter_title = self._id_to_text(title.group(1).strip(), input_json, image_prefix)
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
                    'question': self._id_to_text(q_match.group(1).strip(), input_json, image_prefix) if q_match else "",
                    'answer': a_match.group(1).strip() if a_match else "",
                    'solution': self._id_to_text(s_match.group(1).strip(), input_json, image_prefix) if s_match else "",
                    'label': label,
                    'chapter_title': chapter_title
                })
        return qa_list
    
    def run(self, storage: DataFlowStorage,
            input_response_path_key,
            input_converted_layout_path_key,
            input_name_key,
            output_qalist_path_key,
            ):
        dataframe = storage.read("dataframe")
        
        # Response 转换
        for idx, row in dataframe.iterrows():
            converted_json_path = row[input_converted_layout_path_key]
            response = Path(row[input_response_path_key]).read_text(encoding='utf-8')
            name = row[input_name_key]
            
            image_prefix = os.path.join(name, f"vqa_images")
            qa_list = self._convert_response(response, converted_json_path, image_prefix)
            output_qalist_path = os.path.join(self.output_dir, name, f"extracted_vqa.jsonl")
            os.makedirs(os.path.dirname(output_qalist_path), exist_ok=True)
            with open(output_qalist_path, 'w') as outfile:
                for qa in qa_list:
                    json.dump(qa, outfile, ensure_ascii=False)
                    outfile.write('\n')
            
            # 复制图片
            src_dir = converted_json_path.rpartition('/')[0]
            src_images = os.path.join(src_dir, 'vlm', 'images')
            if not os.path.exists(src_images):
                src_images = os.path.join(src_dir, 'images')
            if not os.path.exists(src_images):
                raise ValueError("Images directory not found! There might be a change in Mineru API!")
            dst_images = os.path.join(self.output_dir, image_prefix)
            
            try:
                if os.path.exists(src_images):
                    shutil.copytree(src_images, dst_images)
                else:
                    self.logger.warning(f"Source images dir does not exist: {src_images}")
            except Exception as e:
                self.logger.warning(f"Failed to copy images from {src_images} to {dst_images}: {e}")
            
            dataframe.loc[idx, output_qalist_path_key] = output_qalist_path
            
        storage.write(dataframe)