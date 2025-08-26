import pandas as pd
from typing import Literal
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import os
from pathlib import Path
from trafilatura import fetch_url, extract
import requests

def _parse_file_with_mineru(raw_file: str, output_file: str, mineru_backend: Literal["vlm-sglang-engine", "pipeline"] = "vlm-sglang-engine") -> str:
    """
    Uses MinerU to parse PDF/image files (pdf/png/jpg/jpeg/webp/gif) into Markdown files.

    Internally, the parsed outputs for each item are stored in a structured directory:
    'intermediate_dir/pdf_name/MinerU_Version[mineru_backend]'.
    This directory stores various MinerU parsing outputs, and you can customize
    which content to extract based on your needs.

    Args:
        raw_file: Input file path, supports .pdf/.png/.jpg/.jpeg/.webp/.gif
        output_file: Full path for the output Markdown file
        mineru_backend: Sets the backend engine for MinerU. Options include:
                        - "pipeline": Traditional pipeline processing (MinerU1)
                        - "vlm-sglang-engine": New engine based on multimodal language models (MinerU2) (default recommended)
                        Choose the appropriate backend based on your needs. Defaults to "vlm-sglang-engine".
                        For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.

    Returns:
        output_file: Path to the Markdown file
    """

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

    logger=get_logger()
    
    os.environ['MINERU_MODEL_SOURCE'] = "local"  # 可选：从本地加载模型

    MinerU_Version = {"pipeline": "auto", "vlm-sglang-engine": "vlm"}

    raw_file = Path(raw_file)
    pdf_name = raw_file.stem
    intermediate_dir = output_file
    try:
        return_code = os.system(
            f"mineru -p {raw_file} -o {intermediate_dir} -b {mineru_backend} --source local"
        )
        if return_code != 0:
            raise RuntimeError(f"MinerU execution failed with return code: {return_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to process file with MinerU: {str(e)}")

    # Directory for storing raw data, including various MinerU parsing outputs.
    # You can customize which content to extract based on your needs.
    PerItemDir = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend])

    output_file = os.path.join(PerItemDir, f"{pdf_name}.md")

    logger.info(f"Markdown saved to: {output_file}")
    return output_file


def _parse_doc_to_md(input_file: str, output_file: str):
    """
        support conversion of doc/ppt/pptx/pdf files to markdowns
    """
    try:
        from magic_doc.docconv import DocConverter
    except:
        raise Exception(
            """
Fairy-doc is not installed in this environment yet.
Please refer to https://github.com/opendatalab/magic-doc to install.
Or you can just execute 'apt-get/yum/brew install libreoffice' and 'pip install fairy-doc[gpu]' to fix this error.
please make sure you have gpu on your machine.
"""
        )
    logger=get_logger()
    converter = DocConverter(s3_config=None)
    markdown_content, time_cost = converter.convert(input_file, conv_timeout=300)
    logger.info("time cost: ", time_cost)
    with open(output_file, "w",encoding='utf-8') as f:
        f.write(markdown_content)
    return output_file

def _parse_xml_to_md(raw_file:str=None, url:str=None, output_file:str=None):
    logger=get_logger()
    if(url):
        downloaded=fetch_url(url)
    elif(raw_file):
        with open(raw_file, "r", encoding='utf-8') as f:
            downloaded=f.read()
    else:
        raise Exception("Please provide at least one of file path and url string.")

    try:
        result=extract(downloaded, output_format="markdown", with_metadata=True)
        logger.info(f"Extracted content is written into {output_file}")
        with open(output_file,"w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        logger.error("Error during extract this file or link: ", e)

    return output_file

def is_pdf_url(url):
    try:
        # 发送HEAD请求，只获取响应头，不下载文件
        response = requests.head(url, allow_redirects=True)
        # 如果响应的Content-Type是application/pdf
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            return True
        else:
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            return False
    except requests.exceptions.RequestException:
        # 如果请求失败，返回False
        print("Request failed")
        return False

def download_pdf(url, save_path):
    try:
        # 发送GET请求下载PDF文件
        response = requests.get(url, stream=True)
        # 确保响应内容是PDF
        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
            # 将PDF保存到本地
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"PDF saved to {save_path}")
        else:
            print("The URL did not return a valid PDF file.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
    
@OPERATOR_REGISTRY.register()
class FileOrURLToMarkdownConverter(OperatorABC):
    """
    mineru_backend sets the backend engine for MinerU. Options include:
    - "pipeline": Traditional pipeline processing (MinerU1)
    - "vlm-sglang-engine": New engine based on multimodal language models (MinerU2) (default recommended)
    Choose the appropriate backend based on your needs.  Defaults to "vlm-sglang-engine".
    For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.
    """
    def __init__(
        self, 
        url: str = None,
        raw_file: str = None,
        intermediate_dir: str = "intermediate", 
        lang: str = "en", 
        mineru_backend:  Literal["vlm-sglang-engine", "pipeline"] = "vlm-sglang-engine",
        ):
        self.logger = get_logger()
        self.intermediate_dir=intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.lang=lang
        self.mineru_backend = mineru_backend
        self.url = url
        self.raw_file = raw_file
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        返回算子功能描述 (根据run()函数的功能实现)
        """
        if lang == "zh":
            return (
                "知识提取算子：支持从多种文件格式中提取结构化内容并转换为标准Markdown\n"
                "核心功能：\n"
                "1. PDF文件：使用MinerU解析引擎提取文本/表格/公式，保留原始布局\n"
                "2. Office文档(DOC/PPT等)：通过DocConverter转换为Markdown格式\n"
                "3. 网页内容(HTML/XML)：使用trafilatura提取正文并转为Markdown\n"
                "4. 纯文本(TXT/MD)：直接透传不做处理\n"
                "特殊处理：\n"
                "- 自动识别中英文文档(lang参数)\n"
                "- 支持本地文件路径和URL输入\n"
                "- 生成中间文件到指定目录(intermediate_dir)"
            )
        else:  # 默认英文
            return (
                "Knowledge Extractor: Converts multiple file formats to structured Markdown\n"
                "Key Features:\n"
                "1. PDF: Uses MinerU engine to extract text/tables/formulas with layout preservation\n"
                "2. Office(DOC/PPT): Converts to Markdown via DocConverter\n"
                "3. Web(HTML/XML): Extracts main content using trafilatura\n"
                "4. Plaintext(TXT/MD): Directly passes through without conversion\n"
                "Special Handling:\n"
                "- Auto-detects Chinese/English documents(lang param)\n"
                "- Supports both local files and URLs\n"
                "- Generates intermediate files to specified directory(intermediate_dir)"
            )

    def run(self, storage: DataFlowStorage, input_key="", output_key=""):
        self.logger.info("Starting extraction...")
        self.logger.info("If you're providing a URL or a large file, this may take a while. Please wait...")

        # Handle extraction from URL
        if self.url:
            if is_pdf_url(self.url):
                pdf_save_path = output_file = os.path.join(
                    os.path.dirname(storage.first_entry_file_name),
                    "raw/crawled.pdf"
                )
                self.logger.info(f"Downloading PDF from {self.url} to {pdf_save_path}")
                download_pdf(self.url, pdf_save_path)
                self.raw_file=pdf_save_path
                self.logger.info(f"pdf file has been fetched and saved to {pdf_save_path}")
            else:       
                output_file = os.path.join(
                    os.path.dirname(storage.first_entry_file_name),
                    "raw/crawled.md"
                )
                output_file = _parse_xml_to_md(url=self.url, output_file=output_file)
                self.logger.info(f"Primary extracted result written to: {output_file}")
                return output_file



        # Handle supported file types
        # Extract file name and extension
        raw_file_name = os.path.splitext(os.path.basename(self.raw_file))[0]
        raw_file_suffix = os.path.splitext(self.raw_file)[1].lower()
        raw_file_suffix_no_dot = raw_file_suffix.lstrip(".")

        # Define default output path
        output_file = os.path.join(
            self.intermediate_dir,
            f"{raw_file_name}_{raw_file_suffix_no_dot}.md"
        )
        if raw_file_suffix in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            self.logger.info(f"Using MinerU backend: {self.mineru_backend}")
            # Use MinerU backend for PDF and image files
            output_file = _parse_file_with_mineru(
                raw_file=self.raw_file,
                output_file=self.intermediate_dir,
                mineru_backend=self.mineru_backend
            )

        elif raw_file_suffix in [".doc", ".docx", ".ppt", ".pptx"]:
            # .doc format is currently not supported
            if raw_file_suffix == ".doc":
                raise Exception(
                    "Function under maintenance. Please convert your file to PDF format first."
                )
            # Handling for .docx, .pptx, and .ppt can be added here if needed

        elif raw_file_suffix in [".html", ".xml"]:
            # Use XML/HTML parser for HTML and XML files
            output_file = _parse_xml_to_md(raw_file=self.raw_file, output_file=output_file)

        elif raw_file_suffix in [".txt", ".md"]:
            # Plain text and markdown files require no processing
            output_file = self.raw_file

        else:
            # Unsupported file type
            raise Exception(f"Unsupported file type: {raw_file_suffix}")

        
        self.logger.info(f"Primary extracted result written to: {output_file}")
        return output_file

