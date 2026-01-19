import os
import json
import re
import pandas as pd
import tiktoken
import shutil
import torch
from pathlib import Path
from typing import Literal
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.core import LLMServingABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt
from dataflow.core.prompt import prompt_restrict
from dataflow.utils.pdf2vqa.format_utils import merge_qa_pair, jsonl_to_md

@prompt_restrict(QAExtractPrompt)
@OPERATOR_REGISTRY.register()
class VQAExtractor(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 mineru_backend: Literal["vlm-transformers","vlm-vllm-engine"] = "vlm-transformers",
                 max_chunk_len: int = 128000,):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = QAExtractPrompt()
        self.mineru_backend = mineru_backend
        self.max_chunk_len = max_chunk_len
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于从试题或图文PDF文档中自动提取问答（VQA）结构化数据。\n\n"
                "功能说明：\n"
                "- 自动调用 MinerU 模型提取 PDF 文档的版面与内容布局信息。\n"
                "- 支持题目与答案的分离提取或交错（interleaved）模式处理。\n"
                "- 基于 LLM 生成章节结构化问答（<chapter>、<qa_pair> 标签格式）。\n"
                "- 自动进行内容清洗、图片路径替换与问答重建。\n"
                "- 支持结果过滤、合并及 Markdown 文档转换。\n\n"
                "输入要求：\n"
                "- DataFrame 中需包含 PDF 文件路径列，可为 question_pdf_path/answer_pdf_path 或 pdf_path。\n\n"
                "初始化参数：\n"
                "- llm_serving: LLM 推理服务实例，用于生成问答。\n"
                "- mineru_backend: MinerU 后端类型（可选值：\"vlm-transformers\" 或 \"vlm-vllm-engine\"）。\n"
                "- max_chunk_len: 单批次最大token数量（默认128000）。\n\n"
                "运行参数（run）：\n"
                "- input_question_pdf_path_key: 题目PDF路径列名（默认：\"question_pdf_path\"）。\n"
                "- input_answer_pdf_path_key: 答案PDF路径列名（默认：\"answer_pdf_path\"）。\n"
                "- input_pdf_path_key: 交错模式下的PDF路径列名（默认：\"pdf_path\"）。\n"
                "- input_subject_key: 学科类别列名（默认：\"subject\"）。\n"
                "- output_dir_key: 输出目录列名（默认：\"output_dir\"）。\n"
                "- output_jsonl_key: 输出JSONL路径列名（默认：\"output_jsonl_path\"）。\n"
                "- output_default_dir: 默认输出目录（默认：\"../vqa_output\"）。\n\n"
                "输出：\n"
                "- 在 DataFrame 中新增一列，记录生成的VQA结构化问答JSONL文件路径。\n"
                "- 同时生成过滤后的Markdown文档和对应图片资源文件夹。"
            )
        elif lang == "en":
            return (
                "This operator extracts structured Visual Question Answering (VQA) data from exam or multimodal PDF documents.\n\n"
                "Functionality:\n"
                "- Automatically uses MinerU models to extract PDF layout and textual content.\n"
                "- Supports both separate (question/answer) and interleaved PDF processing modes.\n"
                "- Generates structured chapter-based QA pairs using an LLM (<chapter>, <qa_pair> tags).\n"
                "- Cleans and reconstructs QA content with proper image references.\n"
                "- Filters, merges, and converts the output into Markdown format.\n\n"
                "Input Requirements:\n"
                "- The input DataFrame must contain PDF path columns: either question_pdf_path/answer_pdf_path or pdf_path.\n\n"
                "Initialization Parameters:\n"
                "- llm_serving: Instance of LLM inference service used for QA generation.\n"
                "- mineru_backend: Backend type for MinerU ('vlm-transformers' or 'vlm-vllm-engine').\n"
                "- max_chunk_len: Maximum number of tokens per batch (default: 128000).\n\n"
                "Run Parameters:\n"
                "- input_question_pdf_path_key: Column name for question PDF path (default: 'question_pdf_path').\n"
                "- input_answer_pdf_path_key: Column name for answer PDF path (default: 'answer_pdf_path').\n"
                "- input_pdf_path_key: Column name for interleaved PDF path (default: 'pdf_path').\n"
                "- input_subject_key: Column name for subject type (default: 'subject').\n"
                "- output_dir_key: Column name for output directory (default: 'output_dir').\n"
                "- output_jsonl_key: Column name for output JSONL file path (default: 'output_jsonl_path').\n"
                "- output_default_dir: Default output directory (default: '../vqa_output').\n\n"
                "Output:\n"
                "- Adds a new column to the DataFrame containing paths to generated structured VQA JSONL files.\n"
                "- Also produces filtered Markdown documents and associated image folders."
            )
        else:
            return "VQAExtractor extracts structured VQA data from PDF documents and outputs filtered JSONL and Markdown files."


    def _convert_json(self, input_file, output_file):
        with open(input_file, 'r') as infile:
            data = list(json.load(infile))
        
        new_data = []
        id = 0
        for item in data:
            item['id'] = id
            item.pop('bbox', None)
            item.pop('page_idx', None)
            if item.get('type','') == 'list':
                if item['sub_type'] == 'text':
                    for idx, list_item in enumerate(item.get('list_items', [])):
                        new_item = {
                            'type': 'text',
                            'text': list_item,
                            'id': id + idx,
                        }
                        new_data.append(new_item)
                    id += len(item.get('list_items', []))
            else:
                new_data.append(item)
                id += 1
        
        with open(output_file, 'w') as outfile:
            json.dump(new_data, outfile, ensure_ascii=False)
    
    def _count_tokens(self, text: str) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    
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
    
    def _extract_doc_layout(self, input_pdf_file_path: str, output_folder: str, mineru_backend: Literal["vlm-transformers","vlm-vllm-engine"] = "vlm-transformers"):
        """提取 PDF 的布局信息（合并自 VQAExtractDocLayoutMinerU）"""
        try:
            import mineru
            from mineru.cli.client import main as mineru_main
        except ImportError:
            raise Exception(
                """
                MinerU is not installed in this environment yet.
                Please refer to https://github.com/opendatalab/mineru to install.
                Or you can just execute 'pip install mineru[pipeline]' and 'mineru-models-download' to fix this error.
                Please make sure you have GPU on your machine.
                """
            )
        try:
            from pypdf import PdfReader, PdfWriter, PageObject
        except ImportError:
            raise Exception(
                """
                pypdf is not installed in this environment yet.
                Please use pip install pypdf.
                """
            )
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            raise Exception(
                """
                reportlab is not installed in this environment yet.
                Please use pip install reportlab.
                """
            )
        
        os.environ['MINERU_MODEL_SOURCE'] = "local"
        
        MinerU_Version = {
            "pipeline": "auto", 
            "vlm-transformers": "vlm", 
            "vlm-vllm-engine": "vlm",
            "vlm-http-client": "vlm",
            "hybrid-http-client": "vlm",
            "vlm-auto-engine": "vlm",
            "hybrid-auto-engine": "vlm"
        }
        
        if mineru_backend == "pipeline":
            raise ValueError("The 'pipeline' backend is not supported due to its incompatible output format. Please use 'vlm-transformers' or 'vlm-vllm-engine' instead.")
        
        raw_file = Path(input_pdf_file_path)
        pdf_name = raw_file.stem
        intermediate_dir = output_folder
        args = [
            "-p", str(raw_file),
            "-o", str(intermediate_dir),
            "-b", mineru_backend,
            "--source", "local"
        ]
        if mineru_backend == "vlm-vllm-engine":
            assert torch.cuda.is_available(), "MinerU vlm-vllm-engine backend requires GPU support."
            args += ["--tensor-parallel-size", "2" if torch.cuda.device_count() >= 2 else "1"]
        
        try:
            mineru_main(args)
        except SystemExit as e:
            if e.code != 0:
                raise RuntimeError(f"MinerU execution failed with exit code: {e.code}")
        
        output_json_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], f"{pdf_name}_content_list.json")
        output_layout_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], f"{pdf_name}_layout.pdf")
        return output_json_file, output_layout_file
    
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
            input_question_pdf_path_key: str = "question_pdf_path",
            input_answer_pdf_path_key: str = "answer_pdf_path",
            input_pdf_path_key: str = "pdf_path",  # 支持 interleaved 模式的单一 pdf_path
            input_subject_key: str = "subject",
            output_dir_key: str = "output_dir",
            output_jsonl_key: str = "output_jsonl_path",
            output_default_dir: str = "../vqa_output") -> list:
        dataframe = storage.read("dataframe")
        
        # 支持两种输入格式：question_pdf_path/answer_pdf_path 或 pdf_path
        if input_question_pdf_path_key not in dataframe.columns and input_pdf_path_key not in dataframe.columns:
            raise ValueError(f"Column '{input_question_pdf_path_key}' or '{input_pdf_path_key}' not found in dataframe")
        
        # ========== Stage 1: 预处理（任务扩展 + Layout 提取） ==========
        expanded_rows = []
        
        for idx, row in dataframe.iterrows():
            # 优先使用 question_pdf_path，如果没有则使用 pdf_path（interleaved 模式）
            if input_question_pdf_path_key in dataframe.columns:
                question_pdf_path = row[input_question_pdf_path_key]
                answer_pdf_path = row.get(input_answer_pdf_path_key, question_pdf_path)
            else:
                # interleaved 模式：使用同一个 pdf_path
                question_pdf_path = row[input_pdf_path_key]
                answer_pdf_path = question_pdf_path
            
            subject = row.get(input_subject_key, "math")
            output_root = row.get(output_dir_key, output_default_dir)
            interleaved = (question_pdf_path == answer_pdf_path)
            
            os.makedirs(output_root, exist_ok=True)
            
            # Question task
            q_outdir = os.path.join(output_root, "question")
            os.makedirs(q_outdir, exist_ok=True)
            
            # Layout 提取
            q_json_path, _ = self._extract_doc_layout(
                input_pdf_file_path=question_pdf_path,
                output_folder=q_outdir,
                mineru_backend=self.mineru_backend
            )
            
            expanded_rows.append({
                "pdf_path": question_pdf_path,
                "mode": "question",
                "interleaved": interleaved,
                "subject": subject,
                "output_dir": q_outdir,
                "output_root": output_root,
                "json_path": q_json_path
            })
            
            # Answer task (if not interleaved)
            if not interleaved:
                a_outdir = os.path.join(output_root, "answer")
                os.makedirs(a_outdir, exist_ok=True)
                
                # Layout 提取
                a_json_path, _ = self._extract_doc_layout(
                    input_pdf_file_path=answer_pdf_path,
                    output_folder=a_outdir,
                    mineru_backend=self.mineru_backend
                )
                
                expanded_rows.append({
                    "pdf_path": answer_pdf_path,
                    "mode": "answer",
                    "interleaved": interleaved,
                    "subject": subject,
                    "output_dir": a_outdir,
                    "output_root": output_root,
                    "json_path": a_json_path
                })
        
        # ========== Stage 2: QA 提取 ==========
        json_paths = [row["json_path"] for row in expanded_rows]
        subjects = [row["subject"] for row in expanded_rows]
        
        user_inputs = []
        split_metadata = []
        
        for idx, input_json_path in enumerate(json_paths):
            subject = subjects[idx] if idx < len(subjects) else subjects[0] if subjects else "math"
            system_prompt = self.prompt_template.build_prompt(subject)
            system_prompt_len = self._count_tokens(system_prompt)
            
            converted_path = input_json_path.replace('.json', '_converted.json')
            self._convert_json(input_json_path, converted_path)
            
            with open(converted_path, 'r') as infile:
                data = json.load(infile)
                assert isinstance(data, list), f"Expected list, got {type(data)} for {input_json_path}"
            
            # 分段处理
            current_chunk, current_len = [], system_prompt_len
            chunks = []
            
            for item in data:
                text = json.dumps(item, ensure_ascii=False)
                item_len = self._count_tokens(text)
                if current_len + item_len > self.max_chunk_len and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk, current_len = [], system_prompt_len
                current_chunk.append(item)
                current_len += item_len
            
            if current_chunk:
                chunks.append(current_chunk)
            
            split_metadata.append(len(chunks))
            
            for chunk in chunks:
                user_inputs.append({
                    'user_input': json.dumps(chunk, ensure_ascii=False),
                    'system_prompt': system_prompt
                })
        
        # 批量生成
        responses = [None] * len(user_inputs)
        current_batch = []
        current_batch_indices = []
        current_system_prompt = None
        
        for idx, item in enumerate(user_inputs):
            user_input = item['user_input']
            system_prompt = item['system_prompt']
            
            if current_system_prompt is None:
                current_system_prompt = system_prompt
                current_batch = [user_input]
                current_batch_indices = [idx]
            elif system_prompt == current_system_prompt:
                current_batch.append(user_input)
                current_batch_indices.append(idx)
            else:
                # 处理当前批次
                batch_responses = self.llm_serving.generate_from_input(user_inputs=current_batch, system_prompt=current_system_prompt)
                for batch_idx, resp in zip(current_batch_indices, batch_responses):
                    responses[batch_idx] = resp
                # 开始新批次
                current_system_prompt = system_prompt
                current_batch = [user_input]
                current_batch_indices = [idx]
        
        # 处理最后一批
        if current_batch:
            batch_responses = self.llm_serving.generate_from_input(user_inputs=current_batch, system_prompt=current_system_prompt)
            for batch_idx, resp in zip(current_batch_indices, batch_responses):
                responses[batch_idx] = resp
        
        # 按 split_metadata 还原
        recombined_responses = []
        idx = 0
        for num_chunks in split_metadata:
            merged_text = "\n".join(responses[idx: idx + num_chunks])
            recombined_responses.append(merged_text)
            idx += num_chunks
        
        # ========== Stage 3: 后处理（Response 转换 + 合并和过滤） ==========
        # Response 转换
        qa_lists = []
        for idx, (response, row) in enumerate(zip(recombined_responses, expanded_rows)):
            json_path = row["json_path"]
            output_dir = row["output_dir"]
            mode = row["mode"]
            output_root = row["output_root"]
            
            image_prefix = f"{mode}_images"
            converted_json_path = json_path.replace('.json', '_converted.json')
            qa_list = self._convert_response(response, converted_json_path, image_prefix)
            
            # 复制图片
            src_dir = os.path.join(output_dir, Path(json_path).stem).replace('_content_list','')
            src_images = os.path.join(src_dir, 'vlm', 'images')
            dst_images = os.path.join(output_root, image_prefix)
            
            try:
                if os.path.exists(src_images):
                    if os.path.exists(dst_images):
                        shutil.rmtree(dst_images)
                    shutil.copytree(src_images, dst_images)
                else:
                    self.logger.warning(f"Source images dir does not exist: {src_images}")
            except Exception as e:
                self.logger.warning(f"Failed to copy images from {src_images} to {dst_images}: {e}")
            
            qa_lists.append(qa_list)
        
        # 按 output_root 分组处理合并和过滤
        output_groups = {}
        for idx, (qa_list, row) in enumerate(zip(qa_lists, expanded_rows)):
            output_root = row["output_root"]
            mode = row["mode"]
            interleaved = row["interleaved"]
            output_dir = row["output_dir"]
            
            if output_root not in output_groups:
                output_groups[output_root] = {
                    "question": None,
                    "answer": None,
                    "interleaved": interleaved
                }
            
            if mode == "question":
                output_groups[output_root]["question"] = (qa_list, output_dir)
            elif mode == "answer":
                output_groups[output_root]["answer"] = (qa_list, output_dir)
        
        # 处理每个 output_root
        result_paths_dict = {}
        for output_root, group_info in output_groups.items():
            q_qa_list, q_output_dir = group_info["question"] if group_info["question"] else (None, None)
            a_qa_list, a_output_dir = group_info["answer"] if group_info["answer"] else (None, None)
            interleaved = group_info["interleaved"]
            
            # 写入 question jsonl (即使为空也要创建文件)
            q_jsonl_path = os.path.join(output_root, "vqa_extracted_questions.jsonl")
            with open(q_jsonl_path, 'w', encoding='utf-8') as f:
                if q_qa_list:
                    for item in q_qa_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 写入 answer jsonl（如果不是 interleaved）
            a_jsonl_path = None
            if not interleaved and a_qa_list:
                a_jsonl_path = os.path.join(output_root, "vqa_extracted_answers.jsonl")
                with open(a_jsonl_path, 'w', encoding='utf-8') as f:
                    for item in a_qa_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 合并
            merged_jsonl = os.path.join(output_root, "vqa_merged_qa_pairs.jsonl")
            if not interleaved and a_jsonl_path:
                merge_qa_pair(q_jsonl_path, a_jsonl_path, merged_jsonl)
            else:
                shutil.copy(q_jsonl_path, merged_jsonl)
            
            # 过滤
            filtered_items = []
            total_count = 0
            with open(merged_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    total_count += 1
                    item = json.loads(line)
                    if item.get('question','').strip() and (item.get('answer','').strip() or item.get('solution','').strip()):
                        filtered_items.append(item)
            
            self.logger.info(f"Before filter: {total_count}, After filter: {len(filtered_items)}")
            
            filtered_jsonl = os.path.join(output_root, "vqa_filtered_qa_pairs.jsonl")
            with open(filtered_jsonl, 'w', encoding='utf-8') as f:
                for item in filtered_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 转换为 markdown
            md_output = os.path.join(output_root, "vqa_filtered_qa_pairs.md")
            jsonl_to_md(filtered_jsonl, md_output)
            
            result_paths_dict[output_root] = filtered_jsonl
        
        # 为原始 dataframe 的每一行分配结果路径
        result_paths = []
        for idx, row in dataframe.iterrows():
            if input_question_pdf_path_key in dataframe.columns:
                question_pdf_path = row[input_question_pdf_path_key]
                answer_pdf_path = row.get(input_answer_pdf_path_key, question_pdf_path)
            else:
                question_pdf_path = row[input_pdf_path_key]
                answer_pdf_path = question_pdf_path
            
            output_root = row.get(output_dir_key, output_default_dir)
            result_paths.append(result_paths_dict.get(output_root))
        
        dataframe[output_jsonl_key] = result_paths
        output_file = storage.write(dataframe)
        self.logger.info(f"VQA extraction complete. Results saved to {output_file}")
        
        return [output_jsonl_key,]

