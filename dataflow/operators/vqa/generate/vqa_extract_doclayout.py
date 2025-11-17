from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import os
import torch
from typing import Literal
from pathlib import Path
        
@OPERATOR_REGISTRY.register()
class VQAExtractDocLayoutMinerU(OperatorABC):
    def __init__(self, mineru_backend: Literal["vlm-transformers","vlm-vllm-engine"] = "vlm-transformers"):
        self.logger = get_logger()
        self.mineru_backend = mineru_backend

    def run(self, storage, input_pdf_file_path:str,
                        output_folder:str):
        global cal_canvas_rect, BlockType, SplitFlag, PdfReader, PdfWriter, PageObject, canvas
        try:
            import mineru
            from mineru.cli.client import main as mineru_main

        except ImportError:
            raise Exception(
            """
            MinerU is not installed in this environment yet.
            Please refer to https://github.com/opendatalab/mineru to install.
            Or you can just execute 'pip install mineru[pipeline]' and 'mineru-models-download' to fix this error.
            Please make sure you have GPU on your machine.
            """
        )
        try:
            from pypdf import PdfReader, PdfWriter, PageObject
        except ImportError:
            raise Exception(
            """
            pypdf is not installed in this environment yet.
            Please use pip install pypdf.
            """
        )
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            raise Exception(
            """
            reportlab is not installed in this environment yet.
            Please use pip install reportlab.
            """
        )

        os.environ['MINERU_MODEL_SOURCE'] = "local"  # 可选：从本地加载模型

        MinerU_Version = {"pipeline": "auto", "vlm-transformers": "vlm", "vlm-vllm-engine": "vlm"}
        
        if self.mineru_backend == "pipeline":
            raise ValueError("The 'pipeline' backend is not supported due to its incompatible output format. Please use 'vlm-transformers' or 'vlm-vllm-engine' instead.")

        raw_file = Path(input_pdf_file_path)
        pdf_name = raw_file.stem
        intermediate_dir = output_folder
        args = [
            "-p", str(raw_file),
            "-o", str(intermediate_dir),
            "-b", self.mineru_backend,
            "--source", "local"
        ]
        if self.mineru_backend == "vlm-vllm-engine":
            assert torch.cuda.is_available(), "MinerU vlm-vllm-engine backend requires GPU support."
            args += ["--tensor-parallel-size", "2" if torch.cuda.device_count() >=2 else "1"] # head是14和16，所以多卡只能2卡

        try:
            mineru_main(args)
        except SystemExit as e:
            # mineru_main 可能会调用 sys.exit()
            if e.code != 0:
                raise RuntimeError(f"MinerU execution failed with exit code: {e.code}")

        output_json_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[self.mineru_backend], f"{pdf_name}_content_list.json")
        output_layout_file = os.path.join(intermediate_dir, pdf_name, MinerU_Version[self.mineru_backend], f"{pdf_name}_layout.pdf")
        return output_json_file, output_layout_file