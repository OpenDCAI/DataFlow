from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.kbc.mineru_api_caller import MinerUBatchExtractorViaAPI
import requests
from urllib.parse import urlparse
import os
from dataflow import get_logger
from trafilatura import fetch_url, extract
from tqdm import tqdm
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from pathlib import Path
from abc import abstractmethod

def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
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
        

    
class MinerUABC(OperatorABC):
    
    def __init__(self, intermediate_dir: str = "intermediate", mineru_backend: str = "vlm-sglang-engine"):
        super().__init__()
        self.logger = get_logger()
        self.intermediate_dir=intermediate_dir
        os.makedirs(self.intermediate_dir, exist_ok=True)
        self.mineru_backend = mineru_backend
        
    def _parse_xml_to_md(self, raw_file:str=None, url:str=None, output_file:str=None):
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

    def _batch_parse_html_or_xml(self, items: list):
        """
        items: List[Dict] with keys:
        - index
        - raw_path or url
        - output_path
        """
        results = {}
        for item in tqdm(items, desc="Parsing HTML/XML", ncols=80):
            try:
                if item.get("url"):
                    out = self._parse_xml_to_md(url=item["url"], output_file=item["output_path"])
                else:
                    out = self._parse_xml_to_md(raw_file=item["raw_path"], output_file=item["output_path"])
                results[item["index"]] = out
            except Exception:
                results[item["index"]] = ""
        return results
    
    def _classify_raw_files(self, storage: DataFlowStorage, df, input_key):
        
        normalized = []
        
        for idx, row in df.iterrows():
            src = row.get(input_key, "")
            item = {"index": idx}

            # URL
            if is_url(src):
                if is_pdf_url(src):
                    pdf_path = os.path.join(
                        os.path.dirname(storage.first_entry_file_name),
                        f"raw/crawled/crawled_{idx}.pdf"
                    )
                    download_pdf(src, pdf_path)
                    item.update({
                        "type": "pdf",
                        "raw_path": pdf_path,
                    })
                else:
                    item.update({
                        "type": "html",
                        "url": src,
                    })

            # local file
            else:
                if not os.path.exists(src):
                    item["type"] = "invalid"
                else:
                    ext = Path(src).suffix.lower()
                    if ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]:
                        item.update({"type": "pdf", "raw_path": src})
                    elif ext in [".html", ".xml"]:
                        item.update({"type": "html", "raw_path": src})
                    elif ext in [".txt", ".md"]:
                        item.update({"type": "text", "raw_path": src})
                    else:
                        item["type"] = "unsupported"

            # output path
            if "raw_path" in item:
                name = Path(item["raw_path"]).stem
                item["output_path"] = os.path.join(self.intermediate_dir, f"{name}.md")
            elif "url" in item:
                item["output_path"] = os.path.join(self.intermediate_dir, f"url_{idx}.md")
                
            normalized.append(item)
    
        pdf_items   = [x for x in normalized if x["type"] == "pdf"]
        html_items  = [x for x in normalized if x["type"] == "html"]
        text_items  = [x for x in normalized if x["type"] == "text"]
        
        return pdf_items, html_items, text_items
    
    @abstractmethod
    def _batch_parse_pdf_with_mineru(self, pdf_files: list):
        pass
    
    def run(self, storage: DataFlowStorage, input_key: str = "source", output_key: str = "text_path"):
        self.logger.info("Starting content extraction (batch mode)...")
        dataframe = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe with {len(dataframe)} entries.")

        pdf_items, html_items, text_items = self._classify_raw_files(storage, dataframe, input_key)

        results = {}

        if html_items:
            results.update(self._batch_parse_html_or_xml(html_items))

        if pdf_items:
            results.update(
                self._batch_parse_pdf_with_mineru(pdf_items)
            )

        for item in text_items:
            results[item["index"]] = item["raw_path"]

        # -------- Stage 4: merge back --------
        dataframe[output_key] = dataframe.index.map(lambda i: results.get(i, ""))

        out_path = storage.write(dataframe)
        self.logger.info(f"Extraction finished. Results saved to {out_path}")
        return out_path
    
    
@OPERATOR_REGISTRY.register()
class FileOrURLToMarkdownConverterAPI(MinerUABC):
    """
    Including mineru via api calling.
    Set your mineru key in `MINERU_API_KEY` environment parameter.
    To get the mineru token, refer to https://mineru.net/apiManage/token.
    For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.
    """
    def __init__(self, intermediate_dir: str = "intermediate", mineru_backend: str = "vlm", api_key:str = None):
        super().__init__(intermediate_dir, mineru_backend)
        if api_key is None:
            try:
                api_key = os.environ["MINERU_API_KEY"]
            except KeyError:
                raise ValueError("MinerU API key not provided. Please set the MINERU_API_KEY environment variable or pass the api_key parameter.")
            
        self.api_key = api_key

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
            
    def _batch_parse_pdf_with_mineru(self, pdf_files: list):
        """
        Batch parse PDFs using MinerU API.

        Args:
            pdf_files (List[Dict]): each item has:
                - index: row index in dataframe
                - raw_path: local pdf path
                - output_path: (ignored, kept for compatibility)
            output_dir (str): base output directory for MinerU results
            mineru_backend (str): MinerU backend name (currently informational)

        Returns:
            Dict[int, str]: mapping from dataframe row index -> markdown path
        """
        import os

        # -------- 1. collect pdf paths --------
        file_paths = [item["raw_path"] for item in pdf_files]

        if not file_paths:
            return {}

        os.makedirs(self.intermediate_dir, exist_ok=True)

        # -------- 2. instantiate MinerU extractor --------
        extractor = MinerUBatchExtractorViaAPI(
            api_key=self.api_key,
            model_version=self.mineru_backend,   # 你现在统一用 vlm
        )

        # -------- 3. run MinerU batch extraction --------
        result = extractor.extract_batch(
            file_paths=file_paths,
            out_dir=self.intermediate_dir,
        )

        # -------- 4. map data_id -> dataframe index --------
        # MinerU data_id 是 enumerate(file_paths) 的顺序字符串
        idx_map = {
            str(i): pdf_files[i]["index"]
            for i in range(len(pdf_files))
        }

        # -------- 5. build final index -> md_path mapping --------
        parsed_results = {}

        for item in result.get("items", []):
            if item.get("state") != "done":
                continue

            data_id = item.get("data_id")
            md_path = item.get("md_path")

            if data_id not in idx_map:
                continue
            if not md_path or not os.path.exists(md_path):
                continue

            parsed_results[idx_map[data_id]] = md_path

        return parsed_results

@OPERATOR_REGISTRY.register()
class FileOrURLToMarkdownConverterBatch(MinerUABC):
    """
    mineru_backend sets the backend engine for MinerU. Options include:
    - "pipeline": Traditional pipeline processing (MinerU1)
    - "vlm-sglang-engine": New engine based on multimodal language models (MinerU2) (default recommended)
    Choose the appropriate backend based on your needs.  Defaults to "vlm-sglang-engine".
    For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.
    """
    def __init__(self, intermediate_dir: str = "intermediate", mineru_backend: str = "vlm"):
        super().__init__(intermediate_dir, mineru_backend)

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

    def _batch_parse_pdf_with_mineru(self, pdf_files: list):
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

        # pipeline|vlm-transformers|vlm-vllm-engine|vlm-http-client
        MinerU_Version = {"pipeline": "auto", "vlm-transformers": "vlm", 'vlm-vllm-engine': 'vlm', 'vlm-http-client': 'vlm'}
        
        parsed_results = {}
        
        for item in pdf_files:
            raw_file = Path(item["raw_path"])
            # import pdb; pdb.set_trace()
            pdf_name = Path(raw_file).stem
            intermediate_dir = self.intermediate_dir
            
            import subprocess

            command = [
                "mineru",
                "-p", raw_file,
                "-o", intermediate_dir,
                "-b", self.mineru_backend,
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

            # PerItemDir = os.path.join(intermediate_dir, pdf_name, MinerU_Version[mineru_backend])
            PerItemDir = os.path.join(intermediate_dir, pdf_name, self.mineru_backend)
            output_file = os.path.join(PerItemDir, f"{pdf_name}.md")
            parsed_results[item["index"]] = output_file
            logger.info(f"Markdown saved to: {output_file}")

        return parsed_results
    
    
@OPERATOR_REGISTRY.register()
class FileOrURLToMarkdownConverterFlash(MinerUABC):

    def __init__(self, intermediate_dir: str = "intermediate", model_path=None, batch_size:int = 4, replicas:int = 1, num_gpus_per_replica:float = 1, engine_gpu_util_rate_to_ray_cap:float = 0.9):
        from flash_mineru import MineruEngine
        
        super().__init__(intermediate_dir, mineru_backend="vlm")
        if model_path is None:
            raise ValueError("Please provide the model_path for MinerUEngine. You can download the model from https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.")
        
        self.flash_mineru_engine = MineruEngine(
            model=model_path, 
            save_dir=intermediate_dir, 
            batch_size=batch_size, 
            replicas=replicas, 
            num_gpus_per_replica=num_gpus_per_replica, engine_gpu_util_rate_to_ray_cap=engine_gpu_util_rate_to_ray_cap
        )

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

    def _batch_parse_pdf_with_mineru(self, pdf_files: list):
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
            import flash_mineru
        except ImportError:
            raise Exception(
                """
                FlashMinerU is not installed in this environment yet.
                Please refer to https://github.com/OpenDCAI/Flash-MinerU to install.
                Or you can just execute 'pip install flash_mineru'.
                Please make sure you have GPU on your machine.
                """
            )

        logger=get_logger()
        pdf_path_list = [item["raw_path"] for item in pdf_files]
        index_name_dict = {Path(item["raw_path"]).stem: item["index"] for item in pdf_files}
        logger.info(f"Running FlashMinerU on {len(pdf_path_list)} files...")
        results = self.flash_mineru_engine.run(pdf_path_list)
        results = [res[0]  for res_list in results for res in res_list]  # flatten [[res]] -> [res]
        parsed_results = {}
        print(results)
        for res in results:
            md_path = Path(res)
            md_path = os.path.abspath(os.path.join(self.intermediate_dir, md_path.stem, 'vlm', md_path.name))
            parsed_results[index_name_dict[Path(md_path).stem]] = md_path
            logger.info(f"Markdown saved to: {md_path}")

        return parsed_results