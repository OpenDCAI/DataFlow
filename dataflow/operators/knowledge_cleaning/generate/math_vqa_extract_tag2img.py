import os
import re
import json
from PIL import Image
import cv2
import logging
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC

class MathVQAExtractTag2Img(OperatorABC):
    def __init__(self, layout_json_dir, pdf_image_dir, output_image_dir, layout_prefix='doclay_page_', image_prefix='page_'):
        """
        初始化处理器。

        Args:
            layout_json_dir (str): 存储布局检测结果的JSON文件的目录。
            pdf_image_dir (str): 存储从PDF转换的原始页面图片的目录。
            output_image_dir (str): 用于保存裁剪出的图片的目录。
        """
        self.layout_json_dir = layout_json_dir
        self.pdf_image_dir = pdf_image_dir
        self.output_image_dir = output_image_dir
        self.layout_prefix = layout_prefix  # 用于处理布局JSON文件的前缀
        self.image_prefix = image_prefix    # 用于处理PDF图片文件的前缀
        
        self.image_counter = 0  # 用于生成唯一的图片文件名
        self.bbox_cache = {}    # 缓存已加载的JSON数据，避免重复读取文件
        self.logger = get_logger()

        # 确保输出目录存在
        os.makedirs(self.output_image_dir, exist_ok=True)
        self.logger.info(f"输出图片目录 '{self.output_image_dir}' 已准备就绪。")

    def _get_bbox(self, page_num, figure_id):
        """
        从对应的JSON文件中获取指定figure的边界框。

        Args:
            page_num (str): 页面编号 (例如, '67')。
            figure_id (str): figure的ID (例如, 'figure2')。

        Returns:
            list or None: 边界框坐标 [x1, y1, x2, y2]，如果未找到则返回 None。
        """
        json_filename = f"{self.layout_prefix}{page_num}.json"
        
        # 检查缓存
        if json_filename not in self.bbox_cache:
            json_path = os.path.join(self.layout_json_dir, json_filename)
            if not os.path.exists(json_path):
                self.logger.warning(f"布局JSON文件未找到: {json_path}")
                return None
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.bbox_cache[json_filename] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"读取或解析JSON文件失败: {json_path}, 错误: {e}")
                return None

        layout_data = self.bbox_cache[json_filename]
        
        # 在detections中查找figure_id
        for detection in layout_data.get("detections", []):
            # class_name 也可以是 'figure'，id 可能是 'figure1', 'figure2' 等
            if detection.get("id") == figure_id:
                return detection.get("bbox")
        
        self.logger.warning(f"在 {json_filename} 中未找到 ID 为 '{figure_id}' 的检测框。")
        return None

    def _replacement_callback(self, match):
        """
        这是 re.sub 的回调函数，用于处理每个匹配到的 <pic> 标签。
        """
        original_tag = match.group(0)
        tag_content = match.group(1) # 例如: "page_67|figure2"

        try:
            page_info, figure_id = tag_content.split('|')
            page_num = page_info.replace('page_', '')
        except ValueError:
            self.logger.error(f"标记格式错误: {original_tag}。将保持原样。")
            return original_tag
        
        # 1. 获取边界框
        bbox = self._get_bbox(page_num, figure_id)
        if not bbox:
            self.logger.warning(f"无法为标记 '{original_tag}' 获取边界框。将保持原样。")
            return original_tag

        # 2. 定位并读取原始图片
        original_image_path = os.path.join(self.pdf_image_dir, f"{self.image_prefix}{page_num}.jpg")
        if not os.path.exists(original_image_path):
            self.logger.warning(f"原始图片文件未找到: {original_image_path}。将保持原样。")
            return original_tag
            
        img = cv2.imread(original_image_path)
        if img is None:
            self.logger.error(f"无法使用OpenCV读取图片: {original_image_path}。")
            return original_tag

        # 3. 裁剪图片
        x1, y1, x2, y2 = map(int, bbox)
        cropped_img = img[y1:y2, x1:x2]
        
        if cropped_img.size == 0:
            self.logger.warning(f"裁剪出的图片为空 (bbox: {bbox})，来自: {original_image_path}。")
            return original_tag

        # 4. 保存裁剪的图片
        self.image_counter += 1
        new_image_name = f"image_{self.image_counter}.jpg"
        output_image_path = os.path.join(self.output_image_dir, new_image_name)
        
        try:
            cv2.imwrite(output_image_path, cropped_img)
        except Exception as e:
            self.logger.error(f"保存裁剪图片失败: {output_image_path}, 错误: {e}")
            return original_tag

        # 5. 生成Markdown链接
        # 使用相对路径，并确保使用正斜杠
        relative_path = f"./{os.path.basename(self.output_image_dir)}/{new_image_name}"
        markdown_link = f"![{tag_content}]({relative_path})"
        
        self.logger.info(f"成功转换标记: {original_tag} -> {markdown_link}")
        return markdown_link

    def process_text(self, text):
        """
        处理单个文本字符串，替换所有 <pic> 标签。
        """
        pattern = re.compile(r"<pic>(.*?)<\/pic>")
        return pattern.sub(self._replacement_callback, text)

    def _dump_markdown(self, processed_qas, output_md_file):
        """
        将 processed_qas 以 Markdown 形式写入到 output_md_file 中，
        格式：
        ## question 1
        ...
        ## answer 1
        ...
        ## question 2
        ...
        ## answer 2
        ...
        """
        try:
            with open(output_md_file, 'w', encoding='utf-8') as f_md:
                for idx, qa in enumerate(processed_qas, 1):
                    question = qa.get('question', '').strip()
                    answer = qa.get('answer', '').strip()
                    f_md.write(f"## question {idx}\n")
                    f_md.write(question + "\n\n")
                    f_md.write(f"## answer {idx}\n")
                    f_md.write(answer + "\n\n")
            self.logger.info(f"Markdown 文件已保存到: {output_md_file}")
        except Exception as e:
            self.logger.error(f"写入 Markdown 文件失败: {output_md_file}, 错误: {e}")

    def run(self, input_qa_file, output_qa_file, output_md_file=None):
        """
        处理包含 QA 对的 JSON Lines 文件，并输出：
        1) 处理后的 JSON Lines 文件 (output_qa_file)
        2) 可选的 Markdown 文件 (output_md_file)，按 ## question i ... ## answer i ... 格式
        """
        processed_qas = []
        # —— 读取 & 处理 JSON Lines —— #
        try:
            with open(input_qa_file, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc="处理QA数据"):
                    try:
                        qa_item = json.loads(line)
                        if 'question' in qa_item and isinstance(qa_item['question'], str):
                            qa_item['question'] = self.process_text(qa_item['question'])
                        if 'answer' in qa_item and isinstance(qa_item['answer'], str):
                            qa_item['answer'] = self.process_text(qa_item['answer'])
                        processed_qas.append(qa_item)
                    except json.JSONDecodeError:
                        self.logger.error(f"跳过无效的JSON行: {line.strip()}")
        except FileNotFoundError:
            self.logger.error(f"输入QA文件未找到: {input_qa_file}")
            return

        # —— 写回 JSON Lines —— #
        try:
            with open(output_qa_file, 'w', encoding='utf-8') as f_out:
                for qa_item in processed_qas:
                    f_out.write(json.dumps(qa_item, ensure_ascii=False) + '\n')
            self.logger.info(f"处理完成！结果已保存到: {output_qa_file}")
            self.logger.info(f"共生成 {self.image_counter} 张裁剪图片。")
        except IOError as e:
            self.logger.error(f"写入输出文件失败: {output_qa_file}, 错误: {e}")

        # —— 可选：输出 Markdown —— #
        if output_md_file:
            self._dump_markdown(processed_qas, output_md_file)