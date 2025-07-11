from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow import get_logger

import pandas as pd

@OPERATOR_REGISTRY.register()
class PretrainFormatConverter(OperatorABC):
    def __init__(self):
        self.logger = get_logger()

    def run(self,
            storage: DataFlowStorage,
            read_key_question: str = "question",
            read_key_answer: str = "answer",
            output_key: str = "text"
            ):
        self.read_key_question = read_key_question
        self.read_key_answer = read_key_answer
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        
        output_rows = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
        output_1 = []
        
        for row in output_rows:
                cur_q = row.get(self.read_key_question) if row.get(self.read_key_question) is not None else ""
                cur_a = row.get(self.read_key_answer) if row.get(self.read_key_answer) is not None else ""
                output_1.append({
                    "text": cur_q + "\n" + cur_a,
                })

        output_file = storage.write(output_1)
        self.logger.info(f"SFT to PT convertion results saved to {output_file}")
        
        return [read_key_question, read_key_answer, output_key]

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于将SFT格式数据转换为预训练格式。\n\n"
                "输入参数：\n"
                "- read_key_question：问题字段名\n"
                "- read_key_answer：答案字段名\n"
                "- output_key：输出文本字段名\n\n"
                "输出参数：\n"
                "- output_key：输出文本字段名，包含问题和答案的拼接结果\n"
                "- 输出文件：转换后的预训练格式数据文件路径"
            )
        elif lang == "en":
            return (
                "Converts SFT format data to pretraining format.\n\n"
                "Input Parameters:\n"
                "- read_key_question: Question field name\n"
                "- read_key_answer: Answer field name\n"
                "- output_key: Output text field name\n\n"
                "Output Parameters:\n"
                "- output_key: Output text field name containing concatenated question and answer\n"
                "- Output file: Path to pretraining format data file"
            )
        else:
            return "FormatConvert_SFT_to_Pretrain: SFT to Pretraining format converter"