import os
from pypdf import PdfWriter
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class PDF_Merger(OperatorABC):
    def __init__(self, output_dir: str):
        """
        初始化 PDF 合并算子。
        
        :param output_dir: 合并后 PDF 文件的存放根目录
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == 'zh':
            return (
                "PDF 文件合并算子。"
                "输入 PDF 路径列表，按顺序合并为一个 PDF 文件，"
                "并保存到指定目录。"
            )
        else:
            return (
                "PDF merging operator."
                "Takes a list of PDF paths, merges them in order into a single PDF,"
                "and saves it to the specified directory."
            )

    def run(self, 
            storage: DataFlowStorage,
            input_pdf_list_key: str,
            input_name_key: str,
            output_pdf_path_key: str
            ):
        """
        执行合并逻辑。
        
        :param input_pdf_list_key: DataFrame 中存放 PDF 路径列表 (str或list[str]) 的列名
        :param input_name_key: DataFrame 中用于命名的列名（如文件名或ID）
        :param output_pdf_path_key: 合并后结果路径存入的列名
        """
        dataframe = storage.read("dataframe")

        for idx, row in dataframe.iterrows():
            pdf_paths = row[input_pdf_list_key]
            if isinstance(pdf_paths, str):
                pdf_paths = [pdf_paths]
            name = row[input_name_key]
            
            # 构建输出路径：output_dir/name/merged.pdf
            save_dir = os.path.join(self.output_dir, str(name))
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, f"{name}_merged.pdf")

            try:
                merger = PdfWriter()
                valid_count = 0
                
                for path in pdf_paths:
                    if os.path.exists(path):
                        merger.append(path)
                        valid_count += 1
                
                if valid_count > 0:
                    with open(output_path, "wb") as f:
                        merger.write(f)
                    merger.close()
                    
                    # 将结果写回 dataframe
                    dataframe.loc[idx, output_pdf_path_key] = output_path
                else:
                    dataframe.loc[idx, output_pdf_path_key] = None
                    
            except Exception as e:
                print(f"Error merging PDFs for {name}: {e}")
                dataframe.loc[idx, output_pdf_path_key] = None

        storage.write(dataframe)