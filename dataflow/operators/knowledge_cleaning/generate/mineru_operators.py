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
        if api_key is None or api_key.strip() == "":
            try:
                api_key = os.environ["MINERU_API_KEY"]
            except KeyError:
                raise ValueError("MinerU API key not provided. Please set the MINERU_API_KEY environment variable or pass the api_key parameter. To get a MinerU API key, visit https://mineru.net/apiManage/docs .")
            
        self.api_key = api_key

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "文件/URL 转 Markdown（MinerU API 版）：批量将输入源转换为 Markdown，并把生成的 Markdown 路径回写到 DataFrame。\n\n"
                "功能说明：\n"
                "1) 自动识别输入类型（本地路径或 URL）：\n"
                "   - URL：若为 PDF（Content-Type=application/pdf）会先下载到 raw/crawled/ 下再解析；否则按网页(HTML/XML)抓取正文转 Markdown。\n"
                "   - 本地文件：支持 PDF/图片(png/jpg/jpeg/webp/gif)、HTML/XML、TXT/MD；TXT/MD 直接透传。\n"
                "2) PDF/图片：通过 MinerU 官方 API 批量解析，输出 Markdown 到 intermediate_dir。\n"
                "3) HTML/XML：使用 trafilatura 抽取正文并输出 Markdown 到 intermediate_dir。\n\n"
                "初始化参数（__init__）：\n"
                "- intermediate_dir: 中间产物目录，保存生成的 Markdown（默认 intermediate）\n"
                "- mineru_backend: MinerU API 的 model_version（默认 vlm；可按后端/版本配置）\n"
                "- api_key: MinerU API Key；不传则读取环境变量 MINERU_API_KEY\n\n"
                "运行参数（run）：\n"
                "- storage: DataFlowStorage，需包含 dataframe\n"
                "- input_key: 输入字段名，单元格为本地路径或 URL（默认 source）\n"
                "- output_key: 输出字段名，写入对应行的 Markdown 文件路径（默认 text_path）\n\n"
                "输出：\n"
                "- 返回 storage.write(...) 的结果路径；并在 dataframe[output_key] 写入解析后的 Markdown 路径（失败为空字符串）。"
            )
        else:
            return (
                "File/URL to Markdown (MinerU API): batch converts sources to Markdown and writes Markdown paths back to the DataFrame.\n\n"
                "What it does:\n"
                "1) Auto-detects input type (local path or URL).\n"
                "   - URL: if it is a PDF (Content-Type=application/pdf), it downloads to raw/crawled/ then parses; otherwise treats it as HTML/XML and extracts main content.\n"
                "   - Local: supports PDF/images (png/jpg/jpeg/webp/gif), HTML/XML, TXT/MD; TXT/MD is passed through.\n"
                "2) PDF/images: parsed in batch via MinerU official API, Markdown saved under intermediate_dir.\n"
                "3) HTML/XML: main content extracted by trafilatura and saved as Markdown under intermediate_dir.\n\n"
                "__init__ params:\n"
                "- intermediate_dir: directory to store generated Markdown\n"
                "- mineru_backend: MinerU API model_version\n"
                "- api_key: MinerU API key (fallback to env MINERU_API_KEY)\n\n"
                "run params:\n"
                "- storage: DataFlowStorage containing a dataframe\n"
                "- input_key: column holding local paths or URLs (default: source)\n"
                "- output_key: column to store generated Markdown file paths (default: text_path)\n\n"
                "Output:\n"
                "- Returns the written dataframe path; dataframe[output_key] contains Markdown paths (empty on failure)."
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
class FileOrURLToMarkdownConverterLocal(MinerUABC):
    """
    mineru_backend sets the backend engine for MinerU. Options include:
    - "pipeline": Traditional pipeline processing (MinerU1)
    - "vlm-sglang-engine": New engine based on multimodal language models (MinerU2) (default recommended)
    Choose the appropriate backend based on your needs.  Defaults to "vlm-sglang-engine".
    For more details, refer to the MinerU GitHub: https://github.com/opendatalab/MinerU.
    """
    def __init__(self, 
                 intermediate_dir: str = "intermediate", 
                 mineru_backend: str = "vlm-auto-engine",
                 mineru_source: str = "local",
                 mienru_model_path:str = None,
                 mineru_download_model_type:str = "vlm"
                 ):
        """
        For mineru_* configeration, please refer to official https://opendatalab.github.io/MinerU/usage/model_source/
            - mineru_backend: backend type. check the `-b` section with `mineru --help` 
            - mineru_source: source type. check `--source` section with `mineru --help` 
            - mineru_model_path: where you put mineru model. check https://opendatalab.github.io/MinerU/usage/model_source/#1-download-models-to-local-storage
            - mineru_download_model_type: which type of mineru you need to download. check `mineru-models-download --help`
        """
        super().__init__(intermediate_dir, mineru_backend)
        self.mineru_source = mineru_source
        self.mienru_model_path = mienru_model_path
        self.mineru_download_model_type = mineru_download_model_type

    @staticmethod
    def get_desc(lang: str = "zh"):
        """
        返回算子功能描述 (根据run()函数的功能实现)
        """
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "文件/URL 转 Markdown（MinerU 本地批处理版）：批量将输入源转换为 Markdown，并把生成的 Markdown 路径回写到 DataFrame。\n\n"
                "功能说明：\n"
                "1) 自动识别输入类型（本地路径或 URL）：\n"
                "   - URL：若为 PDF（Content-Type=application/pdf）会先下载到 raw/crawled/ 下再解析；否则按网页(HTML/XML)抓取正文转 Markdown。\n"
                "   - 本地文件：支持 PDF/图片(png/jpg/jpeg/webp/gif)、HTML/XML、TXT/MD；TXT/MD 直接透传。\n"
                "2) PDF/图片：通过本地 MinerU CLI（subprocess 调用 `mineru`）逐文件解析。\n"
                "   - 输出目录结构：{intermediate_dir}/{pdf_name}/{mineru_backend}/\n"
                "   - 默认 Markdown 路径：{intermediate_dir}/{pdf_name}/{mineru_backend}/{pdf_name}.md\n"
                "3) HTML/XML：使用 trafilatura 抽取正文并输出 Markdown 到 intermediate_dir。\n\n"
                "初始化参数（__init__）：\n"
                "- intermediate_dir: 中间产物目录（默认 intermediate）\n"
                "- mineru_backend: MinerU CLI 后端（默认 vlm-auto-engine；也可 pipeline / vlm-sglang-engine 等）\n"
                "- mineru_source: 模型来源（默认 local；对应 MINERU_MODEL_SOURCE）\n"
                "- mienru_model_path: 本地模型目录；提供则会调用 configure_model 配置模型\n"
                "- mineru_download_model_type: 配置模型类型（默认 vlm）\n\n"
                "运行参数（run）：\n"
                "- storage: DataFlowStorage，需包含 dataframe\n"
                "- input_key: 输入字段名（默认 source），值为本地路径或 URL\n"
                "- output_key: 输出字段名（默认 text_path），写入对应行的 Markdown 文件路径\n\n"
                "环境依赖：\n"
                "- 需安装 mineru 并具备可用的运行环境（通常需要 GPU/模型文件）；未安装会直接抛错。\n\n"
                "输出：\n"
                "- 返回 storage.write(...) 的结果路径；并在 dataframe[output_key] 写入解析后的 Markdown 路径（失败为空字符串）。"
            )
        else:
            return (
                "File/URL to Markdown (MinerU Local Batch): batch converts sources to Markdown and writes Markdown paths back to the DataFrame.\n\n"
                "What it does:\n"
                "1) Auto-detects input type (local path or URL).\n"
                "   - URL: if it is a PDF (Content-Type=application/pdf), it downloads to raw/crawled/ then parses; otherwise treats it as HTML/XML.\n"
                "   - Local: supports PDF/images, HTML/XML, TXT/MD; TXT/MD is passed through.\n"
                "2) PDF/images: parsed via local MinerU CLI (`mineru`) per file.\n"
                "   - Output layout: {intermediate_dir}/{pdf_name}/{mineru_backend}/\n"
                "   - Default Markdown: {intermediate_dir}/{pdf_name}/{mineru_backend}/{pdf_name}.md\n"
                "3) HTML/XML: extracted by trafilatura and saved as Markdown under intermediate_dir.\n\n"
                "__init__ params:\n"
                "- intermediate_dir: directory to store generated artifacts\n"
                "- mineru_backend: MinerU CLI backend (e.g., vlm-auto-engine / pipeline / vlm-sglang-engine)\n"
                "- mineru_source: model source (default local)\n"
                "- mienru_model_path: local model path; if provided, configures models via configure_model\n"
                "- mineru_download_model_type: model type for configuration\n\n"
                "run params:\n"
                "- storage: DataFlowStorage containing a dataframe\n"
                "- input_key: column holding local paths or URLs\n"
                "- output_key: column to store Markdown file paths\n\n"
                "Dependencies:\n"
                "- Requires MinerU installed and properly configured (often GPU/models).\n\n"
                "Output:\n"
                "- Returns the written dataframe path; dataframe[output_key] contains Markdown paths (empty on failure)."
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
        from mineru.cli.models_download import configure_model

        os.environ.setdefault("MINERU_MODEL_SOURCE", self.mineru_source)

        # load local model and config corresponding files https://github.com/opendatalab/MinerU/blob/a12610fb3e9e24488fe3e76cd233ba88ec64bbaf/mineru/cli/models_download.py#L19
        if self.mienru_model_path != None:
            configure_model(self.mienru_model_path, self.mineru_download_model_type)

        parsed_results = {}
        for item in pdf_files:
            raw_file = Path(item["raw_path"])
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

    def __init__(self, intermediate_dir: str = "intermediate", mineru_model_path=None, batch_size:int = 4, replicas:int = 1, num_gpus_per_replica:float = 1, engine_gpu_util_rate_to_ray_cap:float = 0.9):
        try:
            from flash_mineru import MineruEngine
        except ImportError:
            raise Exception(
                """
                FlashMinerU is not installed in this environment yet.
                Please refer to https://github.com/OpenDCAI/Flash-MinerU to install.
                Or you can just execute 'pip install flash_mineru'.
                Please make sure you have GPU on your machine.
                """
            )
        except Exception as e:
            raise RuntimeError(f"Failed to import MineruEngine: {e}") from e

        super().__init__(intermediate_dir, mineru_backend="vlm")
        if mineru_model_path is None:
            raise ValueError("Please provide the model_path for MinerUEngine. You can download the model from https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.")
        
        self.flash_mineru_engine = MineruEngine(
            model=mineru_model_path, 
            save_dir=intermediate_dir, 
            batch_size=batch_size, 
            replicas=replicas, 
            num_gpus_per_replica=num_gpus_per_replica, engine_gpu_util_rate_to_ray_cap=engine_gpu_util_rate_to_ray_cap
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "文件/URL 转 Markdown（FlashMinerU 加速版）：使用 FlashMinerU 引擎批量解析 PDF/图片为 Markdown，并把生成的 Markdown 路径回写到 DataFrame。\n\n"
                "功能说明：\n"
                "1) 自动识别输入类型（本地路径或 URL）：\n"
                "   - URL：若为 PDF（Content-Type=application/pdf）会先下载到 raw/crawled/ 下再解析；否则按网页(HTML/XML)抓取正文转 Markdown。\n"
                "   - 本地文件：支持 PDF/图片(png/jpg/jpeg/webp/gif)、HTML/XML、TXT/MD；TXT/MD 直接透传。\n"
                "2) PDF/图片：调用 FlashMinerU 的 MineruEngine 批量运行解析。\n"
                "   - 解析后 Markdown 统一落盘到 intermediate_dir，并按 FlashMinerU 的目录结构组织。\n"
                "   - 本实现会将引擎返回的 md 文件名映射回对应 dataframe 行，并构造最终 md 绝对路径。\n"
                "3) HTML/XML：使用 trafilatura 抽取正文并输出 Markdown 到 intermediate_dir。\n\n"
                "初始化参数（__init__）：\n"
                "- intermediate_dir: 中间产物目录（默认 intermediate）\n"
                "- mineru_model_path: FlashMinerU 使用的模型路径（必填；如 MinerU2.5-xxx 权重目录）\n"
                "- batch_size: 批处理大小（默认 4）\n"
                "- replicas: 引擎副本数（默认 1）\n"
                "- num_gpus_per_replica: 每个副本占用 GPU 数（默认 1）\n"
                "- engine_gpu_util_rate_to_ray_cap: Ray 资源利用率上限系数（默认 0.9）\n\n"
                "运行参数（run）：\n"
                "- storage: DataFlowStorage，需包含 dataframe\n"
                "- input_key: 输入字段名（默认 source），值为本地路径或 URL\n"
                "- output_key: 输出字段名（默认 text_path），写入对应行的 Markdown 文件路径\n\n"
                "环境依赖：\n"
                "- 需安装 flash_mineru 并具备可用 GPU/模型文件；未安装会直接抛错。\n\n"
                "输出：\n"
                "- 返回 storage.write(...) 的结果路径；并在 dataframe[output_key] 写入解析后的 Markdown 路径（失败为空字符串）。"
            )
        else:
            return (
                "File/URL to Markdown (FlashMinerU): uses FlashMinerU to batch-parse PDF/images into Markdown and writes Markdown paths back to the DataFrame.\n\n"
                "What it does:\n"
                "1) Auto-detects input type (local path or URL).\n"
                "   - URL: if it is a PDF, downloads then parses; otherwise treats it as HTML/XML.\n"
                "   - Local: supports PDF/images, HTML/XML, TXT/MD; TXT/MD is passed through.\n"
                "2) PDF/images: runs FlashMinerU MineruEngine in batch and maps produced Markdown back to dataframe rows.\n"
                "3) HTML/XML: extracted by trafilatura and saved as Markdown under intermediate_dir.\n\n"
                "__init__ params:\n"
                "- intermediate_dir: directory to store generated artifacts\n"
                "- mineru_model_path: model path for FlashMinerU (required)\n"
                "- batch_size, replicas, num_gpus_per_replica, engine_gpu_util_rate_to_ray_cap: engine tuning params\n\n"
                "run params:\n"
                "- storage: DataFlowStorage containing a dataframe\n"
                "- input_key: column holding local paths or URLs\n"
                "- output_key: column to store Markdown file paths\n\n"
                "Dependencies:\n"
                "- Requires flash_mineru installed and GPU/models available.\n\n"
                "Output:\n"
                "- Returns the written dataframe path; dataframe[output_key] contains Markdown paths (empty on failure)."
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
        final_results = []
        # [['bitter_lesson.md', 'crawled_2.md', 'crawled_3.md'], [xxx]] to ['bitter_lesson.md', 'crawled_2.md', 'crawled_3.md']
        for res in results:
            final_results.extend(res)
        # results = [res[0]  for res_list in results for res in res_list]  # flatten [[res]] -> [res]
        parsed_results = {}
        print(final_results)
        for res in final_results:
            md_path = Path(res)
            md_path = os.path.abspath(os.path.join(self.intermediate_dir, md_path.stem, 'vlm', md_path.name))
            parsed_results[index_name_dict[Path(md_path).stem]] = md_path
            logger.info(f"Markdown saved to: {md_path}")

        return parsed_results