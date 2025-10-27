from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import os
import cv2
import json
import math
import torch
import multiprocessing
from collections import defaultdict
# from doclayout_yolo import YOLOv10
from typing import List, Literal
from pathlib import Path

from io import BytesIO

def modified_draw_bbox_with_number(i, bbox_list, page, c, rgb_config, fill_config, draw_bbox=True):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]
    # 强制转换为 float
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])

    for j, bbox in enumerate(page_data):
        # 确保bbox的每个元素都是float
        rect = cal_canvas_rect(page, bbox)  # Define the rectangle  
        
        if draw_bbox:
            if fill_config:
                c.setFillColorRGB(*new_rgb, 0.3)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
            else:
                c.setStrokeColorRGB(*new_rgb)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
        c.setFillColorRGB(*new_rgb, 1.0)
        c.setFontSize(size=10)
        
        c.saveState()
        rotation_obj = page.get("/Rotate", 0)
        try:
            rotation = int(rotation_obj) % 360  # cast rotation to int to handle IndirectObject
        except (ValueError, TypeError):
            logger = get_logger()
            logger.warning(f"Invalid /Rotate value: {rotation_obj!r}, defaulting to 0")
            rotation = 0

        if rotation == 0:
            c.translate(rect[0] + rect[2] + 2, rect[1] + rect[3] - 10)
        elif rotation == 90:
            c.translate(rect[0] + 10, rect[1] + rect[3] + 2)
        elif rotation == 180:
            c.translate(rect[0] - 2, rect[1] + 10)
        elif rotation == 270:
            c.translate(rect[0] + rect[2] - 10, rect[1] - 2)
            
        c.rotate(rotation)
        c.drawString(0, 0, f"tag{i}:")
        c.drawString(0, -10, f"box{j}")
        c.restoreState()

    return c

def modified_draw_bbox_without_number(i, bbox_list, page, c, rgb_config, fill_config):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]

    for bbox in page_data:
        rect = cal_canvas_rect(page, bbox)  # Define the rectangle  

        c.setStrokeColorRGB(new_rgb[0], new_rgb[1], new_rgb[2])
        c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
    return c

def modified_draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    dropped_bbox_list = []
    tables_body_list, tables_caption_list, tables_footnote_list = [], [], []
    imgs_body_list, imgs_caption_list, imgs_footnote_list = [], [], []
    codes_body_list, codes_caption_list = [], []
    titles_list = []
    texts_list = []
    interequations_list = []
    lists_list = []
    list_items_list = []
    indexs_list = []

    for page in pdf_info:
        page_dropped_list = []
        tables_body, tables_caption, tables_footnote = [], [], []
        imgs_body, imgs_caption, imgs_footnote = [], [], []
        codes_body, codes_caption = [], []
        titles = []
        texts = []
        interequations = []
        lists = []
        list_items = []
        indices = []

        for dropped_bbox in page['discarded_blocks']:
            page_dropped_list.append(dropped_bbox['bbox'])
        dropped_bbox_list.append(page_dropped_list)
        for block in page["para_blocks"]:
            bbox = block["bbox"]
            if block["type"] == BlockType.TABLE:
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.TABLE_BODY:
                        tables_body.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_CAPTION:
                        tables_caption.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_FOOTNOTE:
                        if nested_block.get(SplitFlag.CROSS_PAGE, False):
                            continue
                        tables_footnote.append(bbox)
            elif block["type"] == BlockType.IMAGE:
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.IMAGE_BODY:
                        imgs_body.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_CAPTION:
                        imgs_caption.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_FOOTNOTE:
                        imgs_footnote.append(bbox)
            elif block["type"] == BlockType.CODE:
                for nested_block in block["blocks"]:
                    if nested_block["type"] == BlockType.CODE_BODY:
                        bbox = nested_block["bbox"]
                        codes_body.append(bbox)
                    elif nested_block["type"] == BlockType.CODE_CAPTION:
                        bbox = nested_block["bbox"]
                        codes_caption.append(bbox)
            elif block["type"] == BlockType.TITLE:
                titles.append(bbox)
            elif block["type"] in [BlockType.TEXT, BlockType.REF_TEXT]:
                texts.append(bbox)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                interequations.append(bbox)
            elif block["type"] == BlockType.LIST:
                lists.append(bbox)
                if "blocks" in block:
                    for sub_block in block["blocks"]:
                        list_items.append(sub_block["bbox"])
            elif block["type"] == BlockType.INDEX:
                indices.append(bbox)

        tables_body_list.append(tables_body)
        # tables_caption_list.append(tables_caption)
        # tables_footnote_list.append(tables_footnote)
        imgs_body_list.append(imgs_body)
        # imgs_caption_list.append(imgs_caption)
        # imgs_footnote_list.append(imgs_footnote)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        lists_list.append(lists)
        list_items_list.append(list_items)
        indexs_list.append(indices)
        codes_body_list.append(codes_body)
        # codes_caption_list.append(codes_caption)

    layout_bbox_list = []

    for page in pdf_info:
        page_block_list = []
        for block in page["para_blocks"]:
            if block["type"] in [
                BlockType.TEXT,
                BlockType.REF_TEXT,
                BlockType.TITLE,
                BlockType.INTERLINE_EQUATION,
                BlockType.LIST,
                BlockType.INDEX,
                BlockType.IMAGE,
                BlockType.TABLE,
                BlockType.CODE,
            ]:
                bbox = block["bbox"]
                page_block_list.append(bbox)

        layout_bbox_list.append(page_block_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        c = modified_draw_bbox_without_number(i, codes_body_list, page, c, [102, 0, 204], True)
        # c = modified_draw_bbox_without_number(i, codes_caption_list, page, c, [204, 153, 255], True)
        c = modified_draw_bbox_without_number(i, dropped_bbox_list, page, c, [158, 158, 158], True)
        c = modified_draw_bbox_without_number(i, tables_body_list, page, c, [204, 204, 0], True)
        # c = modified_draw_bbox_without_number(i, tables_caption_list, page, c, [255, 255, 102], True)
        # c = modified_draw_bbox_without_number(i, tables_footnote_list, page, c, [229, 255, 204], True)
        c = modified_draw_bbox_without_number(i, imgs_body_list, page, c, [153, 255, 51], True)
        # c = modified_draw_bbox_without_number(i, imgs_caption_list, page, c, [102, 178, 255], True)
        # c = modified_draw_bbox_without_number(i, imgs_footnote_list, page, c, [255, 178, 102], True)
        c = modified_draw_bbox_without_number(i, titles_list, page, c, [102, 102, 255], True)
        c = modified_draw_bbox_without_number(i, texts_list, page, c, [153, 0, 76], True)
        c = modified_draw_bbox_without_number(i, interequations_list, page, c, [0, 255, 0], True)
        c = modified_draw_bbox_without_number(i, lists_list, page, c, [40, 169, 92], True)
        c = modified_draw_bbox_without_number(i, list_items_list, page, c, [40, 169, 92], False)
        c = modified_draw_bbox_without_number(i, indexs_list, page, c, [40, 169, 92], True)
        c = modified_draw_bbox_with_number(i, layout_bbox_list, page, c, [255, 0, 0], False, draw_bbox=False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"layout.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # 保存结果
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)
        
@OPERATOR_REGISTRY.register()
class VQAExtractDocLayoutMinerU(OperatorABC):
    def __init__(self, mineru_backend: Literal["vlm-transformers","vlm-vllm-engine"] = "vlm-transformers"):
        self.logger = get_logger()
        self.mineru_backend = mineru_backend

    def run(self, storage, input_pdf_file_path:str,
                        output_folder:str):
        try:
            import mineru
            from mineru.utils.draw_bbox import cal_canvas_rect
            from mineru.utils.enum_class import BlockType, ContentType, SplitFlag
            mineru.utils.draw_bbox.draw_layout_bbox = modified_draw_layout_bbox   # 修改画图逻辑
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

        os.environ['MINERU_MODEL_SOURCE'] = "local"  # 可选：从本地加载模型

        MinerU_Version = {"pipeline": "auto", "vlm-transformers": "vlm", "vlm-vllm-engine": "vllm"}
        
        if self.mineru_backend == "pipeline":
            raise ValueError("The 'pipeline' backend is not supported due to its incompatible output format. Please use 'vlm-transformers' or 'vlm-vllm-engine' instead.")

        raw_file = Path(input_pdf_file_path)
        pdf_name = raw_file.stem
        intermediate_dir = output_folder
        args = [
            "-p", str(raw_file),
            "-o", str(intermediate_dir),
            "-b", self.mineru_backend,
            "--source", "local"
        ]

        try:
            mineru_main(args)
        except SystemExit as e:
            # mineru_main 可能会调用 sys.exit()
            if e.code != 0:
                raise RuntimeError(f"MinerU execution failed with exit code: {e.code}")

        output_json_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[self.mineru_backend], f"{pdf_name}_content_list.json")
        output_layout_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[self.mineru_backend], f"{pdf_name}_layout.pdf")
        return output_json_file, output_layout_file