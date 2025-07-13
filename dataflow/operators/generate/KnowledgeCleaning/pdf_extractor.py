import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
import os
from pathlib import Path
from trafilatura import fetch_url, extract
from urllib.parse import urlparse

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def _parse_pdf_to_md(
    input_pdf_path: str, 
    output_dir: str,      
    lang: str = "ch",     
    parse_method: str = "auto"  # 解析方法：auto/txt/ocr
):
    """
    将PDF转换为Markdown（仅使用Pipeline后端）
    """
    try:
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        from mineru.utils.enum_class import MakeMode
    except:
        raise Exception(
            """
MinerU is not installed in this environment yet.
Please refer to https://github.com/opendatalab/mineru to install.
Or you can just execute 'pip install mineru[pipeline]' and 'mineru-models-download' to fix this error.
please make sure you have gpu on your machine.
"""
        )
    
    logger=get_logger()
    # 读取PDF文件
    pdf_bytes = Path(input_pdf_path).read_bytes()
    pdf_name = Path(input_pdf_path).stem

    # 解析PDF
    infer_results, all_image_lists, all_pdf_docs, _, ocr_enabled_list = pipeline_doc_analyze(
        [pdf_bytes], [lang], parse_method=parse_method
    )

    # 准备输出目录
    image_dir = os.path.join(output_dir, f"{pdf_name}_images")
    os.makedirs(image_dir, exist_ok=True)
    image_writer = FileBasedDataWriter(image_dir)
    md_writer = FileBasedDataWriter(output_dir)

    # 生成中间结果和Markdown
    middle_json = pipeline_result_to_middle_json(
        infer_results[0], all_image_lists[0], all_pdf_docs[0], 
        image_writer, lang, ocr_enabled_list[0], True
    )
    md_content = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, os.path.basename(image_dir))
    # 保存Markdown
    md_writer.write_string(f"{pdf_name}_pdf.md", md_content)
    logger.info(f"Markdown saved to: {os.path.join(output_dir, f'{pdf_name}_pdf.md')}")

    return os.path.join(output_dir,f"{pdf_name}_pdf.md")

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
class PDFExtractor(OperatorABC):
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

    def run(self, storage:DataFlowStorage, input_key: str = "raw_content", output_key: str = "text_path"):
        self.logger.info("starting to extract...")
        self.logger.info("If you are providing a url or a large file, this may take a while, please wait...")
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of pdfs: {len(dataframe)}")
        output_file_all = []

        for index, row in dataframe.iterrows():
            content = row.get(input_key, '')
            if is_url(content):
                # 保存为本地临时文件
                local_file_path = f"./.cache/downloaded_{index}.xml"  # 可换成.pdf/.html等
                try:
                    with open(local_file_path, "w", encoding="utf-8") as f:
                        f.write(fetch_url(content))
                except Exception as e:
                    self.logger.error(f"Failed to fetch URL: {content}, error: {e}")
                    continue

                output_file = storage.first_entry_file_name.replace(".jsonl", f"_md_{index}.md")
                output_file = _parse_xml_to_md(raw_file=local_file_path, output_file=output_file)
                self.logger.info(f"Primary extracted result written to: {output_file}")
                output_file_all.append(output_file)
            else:
                if not os.path.exists(content):
                    output_file=""
                    self.logger.error(f"File Not Found Error: Path {content} does not exist!")
                else:
                    _, ext = os.path.splitext(content)
                    if ext in [".pdf"]:
                        output_file=_parse_pdf_to_md(
                            content,
                            self.intermediate_dir,
                            self.lang,
                            "txt"
                        )
                    #elif ext in [".doc", ".docx", ".pptx", ".ppt"]:
                    elif ext in [".html", ".xml"]:
                        output_file=_parse_xml_to_md(raw_file=content,output_file=output_file)
                    elif ext in [".txt",".md"]:
                        # for .txt and .md file, no action is taken
                        output_file=content
                    else:
                        self.logger.error(f"Type Error: {ext} file is not supported for {content}")
                        output_file = ""
                output_file_all.append(output_file)
        dataframe[output_key] = output_file_all
        output_file_path = storage.write(dataframe)
        self.logger.info(f"Primary extracted result written to: {output_file_path}")
        return output_file_path

