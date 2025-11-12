from dataflow.core import OperatorABC
import fitz
import os
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class VQAExtractPdf2Img(OperatorABC):
    def __init__(self, dpi: int = 300):
        self.logger = get_logger()
        self.dpi = dpi
    def run(self, storage, input_pdf_path: str, output_image_folder: str):
        '''
        用来把pdf文件转换为图片的辅助函数
        输入：
            pdf_path: pdf文件路径
            output_folder: 输出图片文件夹路径
        '''
        doc = fitz.open(input_pdf_path)
        # make output directory if it doesn't exist
        os.makedirs(output_image_folder, exist_ok=True)
        # convert each page to image
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=self.dpi)
            pix.save(f"{output_image_folder}/page_{page_index}.jpg")
            self.logger.info(f"Converted page {page_index} to image")
        return output_image_folder