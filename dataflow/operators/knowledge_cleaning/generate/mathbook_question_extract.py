import sys

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
import os
from pathlib import Path
import json
import shutil
import fitz # pip install pymupdf
from dataflow.prompts.kbcleaning import MathbookQuestionExtractPrompt
import re
from openai import OpenAI
import base64
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.core import LLMServingABC
from dataflow.serving import APIVLMServing_openai



@OPERATOR_REGISTRY.register()
class MathBookQuestionExtract(OperatorABC):
    def __init__(self, llm_serving: APIVLMServing_openai, prompt_template = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = MathbookQuestionExtractPrompt()

    @staticmethod   
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于从数学教材PDF中提取问题和相关图片内容。它将PDF转换为图片，使用MinerU进行内容提取，"
                "然后组织图片并使用大语言模型分析内容，最终生成包含问题和图片的JSON和Markdown文件。\n"
                "输入参数：\n"
                "- llm_serving：VLM服务对象，需实现APIVLMServing_openai接口\n"
                "- pdf_file_path：PDF文件路径\n"
                "- output_file_name：输出文件名\n"
                "- output_folder：输出文件夹路径\n"
                "- MinerU_Backend：MinerU后端类型，默认为'vlm-sglang-engine'\n"
                "- dpi：PDF转图片的分辨率，默认为300\n"
                "- api_url：API服务URL\n"
                "- key_name_of_api_key：API密钥的环境变量名\n"
                "- model_name：使用的模型名称，默认为'o4-mini'\n"
                "- max_workers：最大并行工作线程数，默认为20\n"
                "输出参数：\n"
                "- 返回布尔值表示处理是否成功\n"
                "- 在指定文件夹生成JSON和Markdown格式的提取结果"
            )
        elif lang == "en":
            return (
                "This operator extracts questions and related images from mathematics textbook PDFs. It converts the PDF to images, "
                "uses MinerU for content extraction, organizes the images, and analyzes the content using a large vision-language model, "
                "ultimately generating JSON and Markdown files containing questions and images.\n"
                "Input Parameters:\n"
                "- llm_serving: VLM serving object implementing APIVLMServing_openai interface\n"
                "- pdf_file_path: Path to the PDF file\n"
                "- output_file_name: Name for the output files\n"
                "- output_folder: Path to the output folder\n"
                "- MinerU_Backend: MinerU backend type, default is 'vlm-sglang-engine'\n"
                "- dpi: Resolution for PDF to image conversion, default is 300\n"
                "- api_url: API service URL\n"
                "- key_name_of_api_key: Environment variable name for API key\n"
                "- model_name: Model name to use, default is 'o4-mini'\n"
                "- max_workers: Maximum number of parallel workers, default is 20\n\n"
                "Output Parameters:\n"
                "- Returns boolean indicating success of processing\n"
                "- Generates extraction results in JSON and Markdown formats in the specified folder"
            )
        else:
            return (
                "MathBookQuestionExtract processes mathematics textbook PDFs to extract questions and images using MinerU and VLM."
            )
    
    def mineru2_runner(self,
                        pdf_file_path:str,
                        output_folder:str,
                        mineru_backend: Literal["vlm-sglang-engine", "pipeline"] = "vlm-sglang-engine"
                        ):

        try:
            import mineru
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

        MinerU_Version = {"pipeline": "auto", "vlm-sglang-engine": "vlm"}

        raw_file = Path(pdf_file_path)
        pdf_name = raw_file.stem
        intermediate_dir = output_folder
        try:
            return_code = os.system(
                f"mineru -p {raw_file} -o {intermediate_dir} -b {mineru_backend} --source local"
            )
            if return_code != 0:
                raise RuntimeError(f"MinerU execution failed with return code: {return_code}")
        except Exception as e:
            raise RuntimeError(f"Failed to process file with MinerU: {str(e)}")

        output_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], f"{pdf_name}_content_list.json")
        output_pic_folder = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend], "images")
        self.logger.info(f"MinerU json file has been saved to {output_file}")
        return output_file, output_pic_folder
    
    def organize_pics(
        self,
        mineru_content_json_path: str,
        mineru_image_folder: str,
        output_file_path: str,
        output_pic_folder: str
    ):
        '''
        用来把mineru切割出来的图片组织到最终文件夹下的辅助函数
        输入：
            mineru_content_json_path: mineru切割出来的json文件路径
            mineru_image_folder: mineru切割出来的图片文件夹路径
        输出：
            output_file_path: 组织图片后的图片信息记录文件，服务后续的图片处理
            output_pic_folder: 最终组织后的图片文件夹路径
        '''
        global_counter = 0
        global_json_data = []

        # read mineru content json
        json_data = json.load(open(mineru_content_json_path, 'r'))

        # if output_pic_folder is not exist, create it
        if not os.path.exists(output_pic_folder):
            os.makedirs(output_pic_folder)

        for item in json_data:
            if item['type'] == 'image':
                # get the image name
                image_name = item['img_path'].split('/')[-1]
                # get the image path
                image_path = os.path.join(mineru_image_folder, image_name)
                
                page_idx = item['page_idx']
                
                # rename the image
                new_image_name = f"{global_counter}.jpg"
                new_image_path = os.path.join(output_pic_folder, new_image_name)
                shutil.copy(image_path, new_image_path)

                # add to global json data
                global_json_data.append({
                    "img_path": new_image_path,
                    "page_idx": page_idx,
                })
                global_counter += 1
                
        # write to json file
        with open(output_file_path, 'w') as f:
            json.dump(global_json_data, f, indent=4)


    def pdf2images(self, pdf_path: str, output_folder: str, dpi: int = 300):
        '''
        用来把pdf文件转换为图片的辅助函数
        输入：
            pdf_path: pdf文件路径
            output_folder: 输出图片文件夹路径
        '''
        doc = fitz.open(pdf_path)
        # make output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # convert each page to image
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi)
            pix.save(f"{output_folder}/page_{page_index}.jpg")
            self.logger.info(f"Converted page {page_index} to image")
        return True
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_input(self,
                    page_folder: str,
                    img_json_path: str
                    ):
        # 加载page_folder内所有的page_n.jpg
        page_list = [os.path.join(page_folder, f) for f in os.listdir(page_folder) if f.endswith(('.jpg'))]
        idx_list = [int(f.split("/")[-1].split(".")[0].split("_")[-1]) for f in page_list]
        max_page_idx = max(idx_list)

        # load img_json
        img_json = json.load(open(img_json_path, "r"))
        img_dict = {}
        for item in img_json:
            if item["page_idx"] not in img_dict:
                img_dict[item["page_idx"]] = []
            img_dict[item["page_idx"]].append(item["img_path"])

        full_input_image_list = []
        full_input_label_list = []

        for page_idx in range(max_page_idx):
            image_list = []
            label_list = []
            image_list.append(os.path.join(page_folder, f"page_{page_idx}.jpg"))
            label_list.append(f"page_{page_idx}")
            image_list.append(os.path.join(page_folder, f"page_{page_idx+1}.jpg"))
            label_list.append(f"page_{page_idx+1}")
            
            if page_idx in img_dict:
                image_list.extend(img_dict[page_idx])
                label_list.extend([img_path.split("/")[-1] for img_path in img_dict[page_idx]])
            if page_idx+1 in img_dict:
                image_list.extend(img_dict[page_idx+1])
                label_list.extend([img_path.split("/")[-1] for img_path in img_dict[page_idx+1]])
            full_input_image_list.append(image_list)
            full_input_label_list.append(label_list)
        return full_input_image_list,full_input_label_list

    def analyze_and_save(self,result_list,save_folder,img_folder,output_file_name):
        # make save_folder if not exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # make save_folder/images if not exist
        if not os.path.exists(os.path.join(save_folder, "images")):
            os.makedirs(os.path.join(save_folder, "images"))
        
        output_json = []
        output_markdown_text = ""

        for item in result_list:
            if not item:
                continue
            split_text = item.split("<SPACE>")
            for text in split_text:
                if not text:
                    continue
                # 检查所有形如<image>index.jpg</image>这样的内容，比如<image>1.jpg</image>,严格匹配<image>*.jpg</image>
                pic_list = []
                pic_match = re.findall(r'<image>(.*?)\.jpg</image>', text)
                if pic_match:
                    for pic_name in pic_match:
                        # 传入完整路径
                        pic_list.append(os.path.join(img_folder, f"{pic_name}.jpg"))

                    # 生成json风格tezt：直接删掉所有<image>index.jpg</image>
                    json_text = re.sub(r'<image>(.*?)\.jpg</image>', '', text)

                    # 生成markdown风格text：把<image>index.jpg</image>替换为![index.jpg](img_folder/index.jpg)
                    markdown_text = text
                    for pic_name in pic_match:
                        # 把img_folder/pic_name.jpg copy 到 save_folder/images/pic_name.jpg
                        shutil.copy(os.path.join(img_folder, f"{pic_name}.jpg"), os.path.join(save_folder, "images", f"{pic_name}.jpg"))
                        markdown_text = markdown_text.replace(f"<image>{pic_name}.jpg</image>", f"![](images/{pic_name}.jpg)")
                else:
                    json_text = text
                    markdown_text = text
                    pic_list = []
                json_data = {
                    "text": json_text,
                    "pics": pic_list
                }
                output_json.append(json_data)
                output_markdown_text += markdown_text + "\n" + "---" + "\n"
        # save output_json to save_folder
        with open(os.path.join(save_folder, f"{output_file_name}.json"), "w") as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        # save output_markdown_text to save_folder
        with open(os.path.join(save_folder, f"{output_file_name}.md"), "w", encoding="utf-8") as f:
            f.write(output_markdown_text)
        return output_json,output_markdown_text

    def run(
        self,
        pdf_file_path: str,
        output_file_name: str,
        output_folder: str,
        MinerU_Backend: str = "vlm-sglang-engine",
        dpi: int = 300,
        api_url: str = "http://123.129.219.111:3000/v1",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "o4-mini",
        max_workers: int = 20
    ):
        api_key = os.environ.get(key_name_of_api_key)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {key_name_of_api_key}")
        
        # 1. convert pdf to images
        pdf2images_folder_name = output_folder+"/pdfimages"
        self.pdf2images(pdf_file_path, pdf2images_folder_name, dpi)

        # 2. use mineru to extract content and pics
        json_content_file, pic_folder = self.mineru2_runner(pdf_file_path, output_folder, MinerU_Backend)

        # 3. organize_pics
        output_image_folder = output_folder+"/organized_images"
        output_json_file = output_folder+"/organized_images/organized_info.json"
        self.organize_pics(json_content_file, pic_folder,output_json_file, output_image_folder)

        # 4. process input
        full_input_image_list,full_input_label_list = self.process_input(pdf2images_folder_name, output_json_file)

        # 5. init server and generate
        system_prompt = self.prompt_template.build_prompt()
        result_text_list = self.llm_serving.generate_from_input_multi_images(
            list_of_image_paths=full_input_image_list,
            list_of_image_labels=full_input_label_list,
            system_prompt=system_prompt,
            model=model_name,
            timeout=1800
        )

        # 6. save responses
        self.analyze_and_save(result_text_list, output_folder, output_image_folder, output_file_name)

        # 7. return
        return True