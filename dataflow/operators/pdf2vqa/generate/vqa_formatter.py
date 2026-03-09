import os
import json
import re
from pathlib import Path
from typing import Optional, List
import glob

import pandas as pd

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger


@OPERATOR_REGISTRY.register()
class VQAFormatter(OperatorABC):
    """
    从JSONL格式的VQA数据中提取问答对，转换为ShareGPT多模态微调格式

    Input:  JSONL rows with fields: question, answer/solution, and optional image refs
    Output: ShareGPT-format JSON  with 'messages' and 'images' fields
    """

    # 匹配 Markdown 图片引用，例如 ![图3.13](path/to/img.jpg)
    _IMAGE_PATTERN = re.compile(r'!\[.*?\]\((.*?)\)')

    def __init__(
            self,
            output_json_file: Optional[str] = None,
            image_placeholder: str = "<image>",
    ):
        """
        初始化 VQAFormatter。

        Args:
            output_json_file:         输出 ShareGPT JSON 文件路径（可选；
                                      不指定则只更新 DataFrame，不落盘）。
            image_placeholder:        在消息内容中代替图片的占位符，
                                      默认 '<image>'，与 LLaMA-Factory 保持一致。
        """
        self.logger = get_logger()
        self.output_json_file = output_json_file
        self.image_placeholder = image_placeholder

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "VQA提取器 - 将JSONL格式的视觉问答数据转换为ShareGPT多模态微调格式\n\n"
                "核心功能:\n"
                "从结构化的JSONL数据中提取 question/answer/solution 及内嵌图片路径，\n"
                "输出符合 LLaMA-Factory ShareGPT 标准的 messages + images 格式。\n\n"
                "初始化参数:\n"
                "• output_json_file:       输出 JSON 文件路径（可选，不指定则只更新 DataFrame）\n"
                "• image_placeholder:      图片占位符（默认: '<image>'）\n"
                "运行参数:\n"
                "• input_qa_item_key:     VQAn内容字段（默认: 'vqa_pair'）\n"
                "• output_messages_key:    输出消息列字段名（默认: 'messages'）\n"
                "• output_images_key:      输出图片列字段名（默认: 'images'）\n\n"
                "输出格式 (ShareGPT):\n"
                "  [{'messages': [{'role': 'user', 'content': '<image>...'},\n"
                "                 {'role': 'assistant', 'content': '...'}],\n"
                "    'images': ['path/to/img.jpg']}]\n\n"
                "适用场景: 多模态VQA微调、数学解题模型训练（配合 LLaMA-Factory）"
            )
        else:
            return (
                "VQA Formatter - Convert JSONL VQA data to ShareGPT multimodal fine-tuning format\n\n"
                "Core Function:\n"
                "Extract question/answer/solution and embedded image paths from structured JSONL,\n"
                "output in LLaMA-Factory ShareGPT standard: messages + images format.\n\n"
                "Initialization Parameters:\n"
                "• output_json_file:       Output JSON path (optional, skip to only update DataFrame)\n"
                "• image_placeholder:      Placeholder token for images (default: '<image>')\n"
                "Runtime Parameters:\n"
                "• input_qa_item_key:      VQA content                (default: 'vqa_pair')\n"
                "• output_messages_key:    Output column for messages (default: 'messages')\n"
                "• output_images_key:      Output column for images   (default: 'images')\n\n"
                "Output Format (ShareGPT):\n"
                "  [{'messages': [{'role': 'user', 'content': '<image>...'},\n"
                "                 {'role': 'assistant', 'content': '...'}],\n"
                "    'images': ['path/to/img.jpg']}]\n\n"
                "Use Cases: Multimodal VQA fine-tuning, math reasoning model training (LLaMA-Factory)"
            )

    def _extract_images(self, text: str) -> List[str]:
        """
        从文本中提取所有 Markdown 图片路径。

        例: '![图3.13](vqa1/images/abc.jpg)' -> ['vqa1/images/abc.jpg']
        """
        return self._IMAGE_PATTERN.findall(text or "")

    def _strip_image_tags(self, text: str) -> str:
        """移除文本中的 Markdown 图片标记，保留其余内容（含空白清理）。"""
        cleaned = self._IMAGE_PATTERN.sub("", text or "")
        # 合并多余空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def _build_user_content(self, question: str, images: List[str]) -> str:
        """
        构建 user 消息的 content 字段。

        规则: 每张图片在文本前插入一个 image_placeholder，
        与 LLaMA-Factory 的多图约定保持一致。
        """
        prefix = "".join(self.image_placeholder for _ in images)
        question_clean = self._strip_image_tags(question)
        return f"{prefix}{question_clean}" if prefix else question_clean

    def _build_assistant_content(self, answer: str, solution: str) -> str:
        """
        构建 assistant 消息的 content 字段。
        """
        ans_text = (answer or "").strip()
        # 清除 solution 中的图片标记，因为 ShareGPT 格式图片通常放在 user 端或 images 列表里
        sol_text = self._strip_image_tags(solution)

        if ans_text and sol_text:
            # 无论如何都合并：先放答案，再放解题过程
            return f"{ans_text}\n\n{sol_text}"
        elif ans_text:
            return ans_text
        elif sol_text:
            return sol_text
        else:
            return ""

    def _convert_row(
            self,
            row: dict,
            input_qa_item_key: str,
            key_q: str,
            key_a: str,
            key_s: str,
            key_msg: str,
            key_img: str,
            base_path: str,
            image_index: dict
    ) -> Optional[dict]:
        """
        将单行数据转换为 ShareGPT 格式字典。
        """
        
        data_source = row.get(input_qa_item_key, row) if isinstance(row.get(input_qa_item_key), dict) else row

        question = str(data_source.get(key_q) or "").strip()
        answer   = str(data_source.get(key_a) or "").strip()
        solution = str(data_source.get(key_s) or "").strip()

        if not question:
            return None

        abs_images = []
        raw_images = self._extract_images(question) + self._extract_images(solution)
        
        for img_rel_path in raw_images:
            filename = os.path.basename(img_rel_path)
            
            # 直接从内存索引中取
            if filename in image_index:
                full_path = image_index[filename]
            else:
                full_path = os.path.normpath(os.path.join(base_path, img_rel_path))
                self.logger.warning(f"File not found in index: {filename}")

            abs_images.append(full_path)

        user_content      = self._build_user_content(question, abs_images)
        assistant_content = self._build_assistant_content(answer, solution)

        if not assistant_content:
            self.logger.warning(f"Skipping row with empty answer and solution: {question[:60]}...")
            return None

        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        return {key_msg: messages, key_img: abs_images}

    def run(
            self,
            storage: DataFlowStorage,
            input_qa_item_key: Optional[str] = None,
            output_messages_key: str = "messages",
            output_images_key:   str = "images",
    ) -> List[str]:
        self.logger.info("Start converting VQA data to ShareGPT format")

        current_pwd = os.getcwd() 

        df = storage.read(output_type="dataframe")

        if input_qa_item_key not in df.columns:
            raise KeyError(
                f"Pipeline Error: '{input_qa_item_key}' column not found. "
            )

        self.logger.info("Building image path index...")
        current_pwd = os.getcwd()
        image_index = {}
        for img_path in glob.iglob(os.path.join(current_pwd, "**", "*.*"), recursive=True):
            if any(img_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                filename = os.path.basename(img_path)
                if filename not in image_index:
                    image_index[filename] = os.path.abspath(img_path)
        
        self.logger.info(f"Indexed {len(image_index)} images.")

        results: List[dict] = []
        skipped = 0
        for _, row in df.iterrows():
            converted = self._convert_row(
                row.to_dict(),
                input_qa_item_key=input_qa_item_key,
                key_q="question",
                key_a="answer",
                key_s="solution",
                key_msg=output_messages_key,
                key_img=output_images_key,
                base_path=current_pwd,
                image_index=image_index,
            )
            if converted is not None:
                results.append(converted)
            else:
                skipped += 1

        self.logger.info(
            f"Converted {len(results)} samples, skipped {skipped} invalid rows"
        )

        if not results:
            self.logger.warning("No valid VQA samples found after conversion!")
            return [output_messages_key, output_images_key]

        if self.output_json_file:
            output_path = Path(self.output_json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sharegpt_records = [
                {output_messages_key: r[output_messages_key],
                 output_images_key:   r[output_images_key]}
                for r in results
            ]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sharegpt_records, f, indent=2, ensure_ascii=False)
            self.logger.info(f"ShareGPT JSON saved to {output_path}")

        out_df = pd.DataFrame(results)
        storage.write(out_df)

        return [output_messages_key, output_images_key]