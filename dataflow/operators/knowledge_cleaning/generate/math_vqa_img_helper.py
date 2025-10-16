from dataflow.core import OperatorABC
import cv2 as cv
import os
import json
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class MathVQAClipHeader(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    def run(self, input_image_path: str, input_layout_path:str, output_image_folder: str, input_layout_prefix: str = 'doclay'):
        '''
        Clip the headers and footers of images according to document layouts.
        输入：
            input_image_path: 原始图片路径
            input_layout_path: 图片布局路径 （json格式）
            output_image_folder: 裁剪后图片保存路径
        '''
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        with open(input_layout_path, 'r') as f:
            layout = json.load(f)
        for image_name in os.listdir(input_image_path):
            if not image_name.endswith('.png') and not image_name.endswith('.jpg'):
                continue
            image_path = os.path.join(input_image_path, image_name)
            image_id = image_path.split('/')[-1].split('.')[0].split('_')[-1]
            image = cv.imread(image_path)
            h, w, _ = image.shape
            header_height = h
            footer_height = 0
            for block in layout[int(image_id)]:
                if block['type'] not in ['header', 'footer', 'page_number', 'page_annotation']:
                    header_height = min(header_height, block['bbox'][1])
                    footer_height = max(footer_height, block['bbox'][3])
            cropped_image = image[int(header_height * h):int(footer_height * h), 0:w]
            output_image_path = os.path.join(output_image_folder, image_name)
            cv.imwrite(output_image_path, cropped_image)
            self.logger.info(f"Cropped image saved to {output_image_path}")
            
class MathVQAConcatenateImages(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
    def run(self, input_image_folder: str, output_image_folder: str):
        '''
        Concatenate adjacent images in a folder vertically (according to id).
        输入：
            input_image_folder: 图片文件夹路径
            output_image_folder: 拼接后图片文件夹路径
        '''
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        image_names = [name for name in os.listdir(input_image_folder) if name.endswith('.png') or name.endswith('.jpg')]
        image_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Assuming names are like 'page_1.png', 'page_2.png', etc
        i = 0
        while i + 1 < len(image_names):
            image1_path = os.path.join(input_image_folder, image_names[i])
            image1 = cv.imread(image1_path)
            image2_path = os.path.join(input_image_folder, image_names[i + 1])
            image2 = cv.imread(image2_path)
            # padding to ensure both images have the same width
            if image1.shape[1] != image2.shape[1]:
                max_width = max(image1.shape[1], image2.shape[1])
                image1 = cv.copyMakeBorder(image1, 0, 0, 0, max_width - image1.shape[1], cv.BORDER_CONSTANT, value=(255, 255, 255))
                image2 = cv.copyMakeBorder(image2, 0, 0, 0, max_width - image2.shape[1], cv.BORDER_CONSTANT, value=(255, 255, 255))
            # Concatenate images vertically
            concatenated_image = cv.vconcat([image1, image2])
            output_image_name = f"concatenated_{i}.jpg"
            i += 1
            output_image_path = os.path.join(output_image_folder, output_image_name)
            cv.imwrite(output_image_path, concatenated_image)
            self.logger.info(f"Concatenated image saved to {output_image_path}")
            
if __name__ == "__main__":
    operator = MathVQAClipHeader()
    operator.run(
        input_image_path='../vqa/pdf_images',
        input_layout_path='../vqa/layout_images/json',
        output_image_folder='../vqa/cropped_images'
    )
    
    operator = MathVQAConcatenateImages()
    operator.run(
        input_image_folder='../vqa/cropped_images',
        output_image_folder='../vqa/concatenated_images'
    )
    
        