"""Financial PDF parser operator.

Parses financial report / research note PDF files into plain text
for downstream QA generation. Follows the DataFlow extension
operator conventions (OperatorABC + OPERATOR_REGISTRY.register()).
"""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class FinancialPDFParser(OperatorABC):
    """Extract plain text from financial PDF documents (财报/研报).

    延迟依赖：pdfplumber 优先，回退 pypdf。安装：
    pip install dataflow-finance-qa[pdf]
    """

    def __init__(self, max_pages: int | None = None):
        super().__init__()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.max_pages = max_pages

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "解析金融 PDF（年报/财报/研报）为纯文本，"
                "供后续金融 QA 生成算子使用。支持页数上限控制。"
            )
        return (
            "Parse financial PDFs (annual reports / research notes) "
            "into plain text for downstream financial QA generation."
        )

    def _extract(self, pdf_path: str) -> str:
        pages = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if self.max_pages is not None and i >= self.max_pages:
                        break
                    pages.append(page.extract_text() or "")
        except ImportError:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                if self.max_pages is not None and i >= self.max_pages:
                    break
                pages.append(page.extract_text() or "")
        return "\n".join(p for p in pages if p).strip()

    def run(
        self,
        storage: DataFlowStorage,
        input_pdf_path: str = "pdf_path",
        output_text: str = "text",
    ):
        self.logger.info(f"Running {self.__class__.__name__}...")
        dataframe = storage.read("dataframe")
        texts = []
        for pdf_path in dataframe[input_pdf_path]:
            try:
                texts.append(self._extract(pdf_path))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Failed to parse {pdf_path}: {exc}")
                texts.append("")
        dataframe[output_text] = texts
        storage.write(dataframe)
        return [output_text]
