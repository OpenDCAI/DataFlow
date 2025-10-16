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
from mineru.utils.draw_bbox import cal_canvas_rect

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
        
@OPERATOR_REGISTRY.register()
class VQAExtractDocLayoutMinerU(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    def run(self, pdf_file_path:str,
                        output_folder:str,
                        mineru_backend: Literal["vlm-transformers", "pipeline", "vlm-vllm-engine"] = "vlm-transformers"):
        try:
            import mineru
            mineru.utils.draw_bbox.draw_bbox_with_number = modified_draw_bbox_with_number   # 修改画图逻辑
            mineru.utils.draw_bbox.draw_bbox_without_number = modified_draw_bbox_without_number   # 修改画图逻辑
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


        os.environ['MINERU_MODEL_SOURCE'] = "local"  # 可选：从本地加载模型

        MinerU_Version = {"pipeline": "auto", "vlm-transformers": "vlm", "vlm-vllm-engine": "vllm"}
        
        if mineru_backend == "pipeline":
            raise ValueError("The 'pipeline' backend is not supported due to its incompatible output format. Please use 'vlm-transformers' or 'vlm-vllm-engine' instead.")

        raw_file = Path(pdf_file_path)
        pdf_name = raw_file.stem
        intermediate_dir = output_folder
        args = [
            "-p", str(raw_file),
            "-o", str(intermediate_dir),
            "-b", mineru_backend,
            "--source", "local"
        ]

        try:
            mineru_main(args)
        except SystemExit as e:
            # mineru_main 可能会调用 sys.exit()
            if e.code != 0:
                raise RuntimeError(f"MinerU execution failed with exit code: {e.code}")

        output_json_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], f"{pdf_name}_content_list.json")
        output_layout_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], f"{pdf_name}_layout.pdf")
        return output_json_file, output_layout_file