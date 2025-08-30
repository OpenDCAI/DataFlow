import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import os
from pathlib import Path
from trafilatura import fetch_url, extract
from urllib.parse import urlparse
from tqdm import tqdm
import requests

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def _parse_file_with_mineru(raw_file: str, output_file: str, mineru_backend: str = "vlm-sglang-engine") -> str:
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
    # import pdb; pdb.set_trace()
    pdf_name = Path(raw_file).stem
    intermediate_dir = output_file
    intermediate_dir = os.path.join(intermediate_dir, "mineru")
    
    import subprocess

    command = [
        "mineru",
        "-p", raw_file,
        "-o", intermediate_dir,
        "-b", mineru_backend,
        "--source", "local"
    ]

    try:
        result = subprocess.run(
            command,
            #stdout=subprocess.DEVNULL,  
            #stderr=subprocess.DEVNULL,  
            check=True  
        )
    except Exception as e:
        raise RuntimeError(f"Failed to process file with MinerU: {str(e)}")

    # Directory for storing raw data, including various MinerU parsing outputs.
    # You can customize which content to extract based on your needs.
    PerItemDir = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend])

    output_file = os.path.join(PerItemDir, f"{pdf_name}.md")

    logger.info(f"Markdown saved to: {output_file}")
    return output_file


def _parse_xml_to_md(raw_file:str=None, url:str=None, output_file:str=None):
    logger=get_logger()
    if(url):
        downloaded=fetch_url(url)
        if not downloaded:
            downloaded = "fail to fetch this url. Please check your Internet Connection or URL correctness"
            with open(output_file,"w", encoding="utf-8") as f:
                f.write(downloaded)
            return output_file

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
            pdf_folder = os.path.dirname(save_path)
            os.makedirs(pdf_folder, exist_ok=True)
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
class FileOrURLToMarkdownConverterBatch(OperatorABC):
    """
    mineru_backend sets the backend engine for MinerU. Options include:
    - "pipeline": Traditional pipeline processing (MinerU1)
    - "vlm-sglang-engine": New engine based on multimodal language models (MinerU2) (default recommended)
    Choose the appropriate backend based on your needs.  Defaults to "vlm-sglang-engine".
    For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.
    """
    def __init__(self, intermediate_dir: str = "intermediate", lang: str = "en", mineru_backend: str = "vlm-sglang-engine"):
        self.logger = get_logger()
        self.intermediate_dir=intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.lang=lang
        self.mineru_backend = mineru_backend

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

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "text_path"):
        self.logger.info("Starting content extraction...")
        self.logger.info("If the input is a URL or a large file, this process might take some time. Please wait...")

        dataframe = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe with {len(dataframe)} entries.")

        output_file_all = []

        # Wrap iterrows with tqdm for progress tracking
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="FileOrURLToMarkdownConverter Processing files", ncols=100):
            content = row.get(input_key, "")

            if is_url(content):
                # Case: Input is a URL
                if is_pdf_url(content):
                    pdf_save_path = os.path.join(
                        os.path.dirname(storage.first_entry_file_name),
                        f"raw/crawled/crawled_{index}.pdf"
                    )
                    self.logger.info(f"Downloading PDF from {content} to {pdf_save_path}")
                    download_pdf(content, pdf_save_path)
                    content = pdf_save_path
                    self.logger.info(f"pdf file has been fetched and saved to {pdf_save_path}")
                else:
                    output_file = os.path.join(
                        os.path.dirname(storage.first_entry_file_name),
                        f"raw/crawled/crawled_{index}.md"
                    )
                    os.makedirs(os.path.dirname(output_file),exist_ok=True)
                    output_file = _parse_xml_to_md(url=content, output_file=output_file)
                    self.logger.info(f"Primary extracted result written to: {output_file}")
                    output_file_all.append(output_file)
                    continue
        
            # Extract file name and extension
            raw_file = content
            raw_file_name = os.path.splitext(os.path.basename(raw_file))[0]
            raw_file_suffix = os.path.splitext(raw_file)[1].lower()
            raw_file_suffix_no_dot = raw_file_suffix.lstrip(".")

            # Define default output path
            output_file = os.path.join(
                self.intermediate_dir,
                f"{raw_file_name}_{raw_file_suffix_no_dot}.md"
            )

            # Case: Local file path
            if not os.path.exists(content):
                self.logger.error(f"File not found: Path {content} does not exist.")
                output_file_all.append("")
                continue

            _, ext = os.path.splitext(content)
            ext = ext.lower()

            if ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                self.logger.info(f"Using MinerU backend: {self.mineru_backend}")
                output_file = _parse_file_with_mineru(
                    raw_file=content,
                    output_file=self.intermediate_dir,
                    mineru_backend=self.mineru_backend
                )
            elif ext in [".html", ".xml"]:
                output_file = _parse_xml_to_md(raw_file=content, output_file=output_file)
            elif ext in [".txt", ".md"]:
                output_file = content  # No parsing needed for plain text or Markdown files
            else:
                self.logger.error(f"Unsupported file type: {ext} for file {content}")
                output_file = ""

            output_file_all.append(output_file)

        # Save results back to storage
        dataframe[output_key] = output_file_all
        output_file_path = storage.write(dataframe)

        self.logger.info(f"Final extraction results saved to: {output_file_path}")
        return output_file_path
