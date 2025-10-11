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
from doclayout_yolo import YOLOv10
from typing import List


@OPERATOR_REGISTRY.register()
class MathVQAExtractDocLayout(OperatorABC):
    def __init__(self, model_path: str):
        self.logger = get_logger()
        self.model_path = model_path
        
    def check_overlap(self, rect1, rect2):
        """检查两个矩形是否重叠"""
        return not (rect1[2] < rect2[0] or rect1[0] > rect2[2] or 
                    rect1[3] < rect2[1] or rect1[1] > rect2[3])

    def find_best_label_position(self, x1, y1, x2, y2, text_w, text_h, img_shape, existing_boxes, margin=10):
        """为检测框找到最佳的标签位置"""
        img_h, img_w = img_shape[:2]
        # 候选位置（优先级由上到下）
        candidates = [
            {'x': x1,                'y': y1 - text_h - margin, 'type': 'top'},
            {'x': x2 + margin,       'y': y1 + (y2-y1-text_h)//2, 'type': 'right'},
            {'x': x1,                'y': y2 + text_h + margin, 'type': 'bottom'},
            {'x': x1 - text_w - margin, 'y': y1 + (y2-y1-text_h)//2, 'type': 'left'},
            {'x': x1 + margin,       'y': y1 + text_h + margin,  'type': 'inside'},
        ]
        
        def valid(px, py):
            if px < 0 or px+text_w > img_w or py-text_h < 0 or py > img_h:
                return False
            label_rect = [px, py-text_h, px+text_w, py]
            for box in existing_boxes:
                if self.check_overlap(label_rect, box):
                    return False
            return True

        for c in candidates:
            if valid(c['x'], c['y']):
                return c['x'], c['y'], c['type']
        # fallback
        fx = max(0, min(x1, img_w-text_w))
        fy = max(text_h, y1 - margin)
        return fx, fy, 'fallback'

    def draw_adaptive_label(self, image, x1, y1, x2, y2, text, existing_boxes,
                            font=cv2.FONT_HERSHEY_SIMPLEX, fs=1.0, ft=2):
        """在最佳位置绘制带背景的标签"""
        (tw, th), base = cv2.getTextSize(text, font, fs, ft)
        lx, ly, pos = self.find_best_label_position(x1, y1, x2, y2, tw, th, image.shape, existing_boxes)
        cmap = {
            'top':     (0, 255, 255),
            'right':   (255, 0, 255),
            'bottom':  (0, 255, 255),
            'left':    (255, 255, 0),
            'inside':  (255, 165, 0),
            'fallback': (0, 0, 255),
        }
        color = cmap.get(pos, (0, 255, 255))
        pad = 5
        cv2.rectangle(image,
                    (lx-pad, ly-th-pad),
                    (lx+tw+pad, ly+pad),
                    color, -1)
        cv2.putText(image, text, (lx, ly), font, fs, (0, 0, 0), ft)
        return pos

    def worker_process(self, img_list, output_img_dir, output_json_dir, prefix, gpu_id, imgsz, conf_thres):
        """
        worker 进程：
        - 在指定 gpu_id 上加载一次模型
        - 顺序处理分配到自己的多张图片
        """
        # 设置设备
        if gpu_id != "":
            device_str = f"cuda:{gpu_id}"
            # 注意：在多进程环境中设置CUDA_VISIBLE_DEVICES可能不会按预期工作
            # 更好的做法是在启动进程前设置，或者使用torch.cuda.set_device()
            torch.cuda.set_device(gpu_id)
        else:
            device_str = "cpu"
            
        # 加载模型
        model = YOLOv10(self.model_path)

        for current_img_path in img_list:
            try:
                # 1）推理
                dets = model.predict(current_img_path, imgsz=imgsz, conf=conf_thres, device=device_str)
                # 2）读取原图
                img = cv2.imread(current_img_path)
                if img is None:
                    self.logger.error(f"无法读取图片: {current_img_path}")
                    continue
                    
                result = dets[0]
                boxes = result.boxes
                name_map = result.names

                detections = []
                existing_boxes = []
                # 收集所有 box 坐标
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        existing_boxes.append([int(x1), int(y1), int(x2), int(y2)])

                    # 画框 + 标签
                    cls_count = defaultdict(int)
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        conf = float(box.conf[0].cpu().numpy())
                        cid = int(box.cls[0].cpu().numpy())
                        cname = name_map[cid]
                        cls_count[cname] += 1
                        label = f"{cname}{cls_count[cname]}"

                        # 矩形框
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # 自适应标签
                        pos_type = self.draw_adaptive_label(img, x1, y1, x2, y2, label, existing_boxes,
                                                    fs=0.8, ft=2)
                        detections.append({
                            "id": label,
                            "class_name": cname,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "label_position": pos_type
                        })

                # 3）保存图像与 JSON
                base_name = os.path.splitext(os.path.basename(current_img_path))[0]
                out_img_path = os.path.join(output_img_dir, f"{prefix}_{base_name}.jpg")
                out_json_path = os.path.join(output_json_dir, f"{prefix}_{base_name}.json")
                
                # 确保输出目录存在
                os.makedirs(output_img_dir, exist_ok=True)
                os.makedirs(output_json_dir, exist_ok=True)
                
                # 保存图片
                success = cv2.imwrite(out_img_path, img)
                if not success:
                    self.logger.error(f"保存图片失败: {out_img_path}")
                
                # 保存JSON
                with open(out_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "image_path": current_img_path,
                        "total_detections": len(detections),
                        "detections": detections
                    }, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"处理完成: {current_img_path} -> {out_img_path}, {out_json_path}")
                
            except Exception as e:
                self.logger.error(f"处理图片 {current_img_path} 时出错: {str(e)}")

    def batch_process(self, image_paths, output_folder, output_prefix, imgsz=1024, conf_thres=0.2):
        """
        批量处理接口：
        image_paths: List[str] 待处理图片路径列表
        output_folder: str      输出文件夹
        output_prefix: str      输出图片/JSON 的前缀
        其余参数为可选模型配置
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 创建输出子目录
        img_output_dir = os.path.join(output_folder, "images")
        json_output_dir = os.path.join(output_folder, "json")
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(json_output_dir, exist_ok=True)
        
        # 可用的 GPU 列表
        ngpu = torch.cuda.device_count()
        if ngpu == 0:
            # 如果没有 GPU，则当成 1 个 worker 在 CPU 上跑
            gpu_list = [""]
        else:
            gpu_list = list(range(ngpu))

        # 将 image_paths 均匀切分给每个 GPU/worker
        chunks = []
        n = len(image_paths)
        k = len(gpu_list)
        per = math.ceil(n / k) if k > 0 else n
        for i in range(k):
            start_idx = i * per
            end_idx = min((i + 1) * per, n)
            if start_idx < end_idx:
                chunks.append(image_paths[start_idx:end_idx])

        # 启动多进程
        procs = []
        for i, (gpu_id, img_chunk) in enumerate(zip(gpu_list, chunks)):
            if not img_chunk:
                continue
                
            p = multiprocessing.Process(
                target=self.worker_process,
                args=(img_chunk, img_output_dir, json_output_dir, output_prefix,
                      gpu_id, imgsz, conf_thres)
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            
    def run(self, input_image_folder: str, output_folder: str, output_prefix: str = "doclay"):
        os.makedirs(output_folder, exist_ok=True)
        # 获取所有图片路径,确保扩展名为jpg or png
        image_list = []
        for f in os.listdir(input_image_folder):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_list.append(os.path.join(input_image_folder, f))
        
        # 确保图片路径存在
        valid_image_list = []
        for img_path in image_list:
            if os.path.exists(img_path):
                valid_image_list.append(img_path)
            else:
                self.logger.warning(f"图片路径不存在: {img_path}")
        
        if not valid_image_list:
            self.logger.warning("没有找到有效的图片文件")
            return
            
        # 批量处理
        self.batch_process(
            image_paths=valid_image_list,
            output_folder=output_folder,
            output_prefix=output_prefix,
            imgsz=1024,
            conf_thres=0.2
        )