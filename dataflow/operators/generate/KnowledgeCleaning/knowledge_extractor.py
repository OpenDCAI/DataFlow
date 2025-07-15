import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import os
from pathlib import Path
from trafilatura import fetch_url, extract
from termcolor import cprint

def _parse_file_with_mineru(raw_file: str, output_file: str, MinerU_Backend: str = "vlm-sglang-engine") -> str:

    """
    使用 MinerU 将 PDF/图像类文件（pdf/png/jpg/jpeg/webp/gif）解析为 Markdown 文件。
    
    参数:
        raw_file: 输入文件路径，支持 .pdf/.png/.jpg/.jpeg/.webp/.gif
        output_file: 输出 Markdown 文件的完整路径

    返回:
        output_file: Markdown 文件路径
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
            f"mineru -p {raw_file} -o {intermediate_dir} -b {MinerU_Backend} --source local"
        )
        if return_code != 0:
            raise RuntimeError(f"MinerU execution failed with return code: {return_code}")
    except Exception as e:
        raise RuntimeError(f"Failed to process file with MinerU: {str(e)}")

    output_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[MinerU_Backend], f"{pdf_name}.md")

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

@OPERATOR_REGISTRY.register()
class KnowledgeExtractor(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, intermediate_dir: str = "intermediate", lang: str = "en"):
        self.logger = get_logger()
        self.intermediate_dir=intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.lang=lang
        
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

    def run(self, storage: DataFlowStorage, raw_file=None, url=None):
        self.logger.info("Starting extraction...")
        self.logger.info("If you're providing a URL or a large file, this may take a while. Please wait...")

        # Handle extraction from URL
        if url:
            output_file = os.path.join(
                os.path.dirname(storage.first_entry_file_name),
                "raw/crawled.md"
            )
            output_file = _parse_xml_to_md(url=url, output_file=output_file)
            self.logger.info(f"Primary extracted result written to: {output_file}")
            return output_file

        # Extract file name and extension
        raw_file_name = os.path.splitext(os.path.basename(raw_file))[0]
        raw_file_suffix = os.path.splitext(raw_file)[1].lower()
        raw_file_suffix_no_dot = raw_file_suffix.lstrip(".")

        # Define default output path
        output_file = os.path.join(
            self.intermediate_dir,
            f"{raw_file_name}_{raw_file_suffix_no_dot}.md"
        )

        # Handle supported file types
        if raw_file_suffix in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            # Use MinerU backend for PDF and image files
            output_file = _parse_file_with_mineru(
                raw_file=raw_file,
                output_file=self.intermediate_dir,
                MinerU_Backend="vlm-sglang-engine"
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
            output_file = _parse_xml_to_md(raw_file=raw_file, output_file=output_file)

        elif raw_file_suffix in [".txt", ".md"]:
            # Plain text and markdown files require no processing
            output_file = raw_file

        else:
            # Unsupported file type
            raise Exception(f"Unsupported file type: {raw_file_suffix}")

        
        self.logger.info(f"Primary extracted result written to: {output_file}")
        return output_file

