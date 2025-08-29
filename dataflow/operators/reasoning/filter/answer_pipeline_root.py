from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage

import pandas as pd

@OPERATOR_REGISTRY.register()
class AnswerPipelineRoot(OperatorABC):
    def __init__(self):

        self.logger = get_logger()
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "答案处理流程根节点，负责将输入数据根据有无真实标签GT分发到不同处理分支。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_dir：输出目录路径\n"
                "- branch_config：分支配置参数\n"
                "- parallel_workers：并行工作线程数\n\n"
                "输出参数：\n"
                "- 多个输出文件路径（根据分支配置生成）"
            )
        elif lang == "en":
            return (
                "Root node of answer processing pipeline, distributes input data to different processing branches.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_dir: Output directory path\n"
                "- branch_config: Branch configuration parameters\n"
                "- parallel_workers: Number of parallel workers\n\n"
                "Output Parameters:\n"
                "- Multiple output file paths (generated based on branch config)"
            )
        else:
            return "AnswerPipelineRoot routes data to different processing branches."

    def run(self, storage: DataFlowStorage, input_answer_key: str = "output", input_gt_key: str = "golden_answer"):
        self.input_answer_key = input_answer_key
        self.input_gt_key = input_gt_key
        
        df = storage.read("dataframe")

        if not self.input_gt_key or self.input_gt_key not in df.columns:
            self.logger.warning("No valid gt key in input file, copy input file to output file without gt")
            return
            
        # 初始化答案提取器
        if self.input_answer_key in df.columns:
            unit_text_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_text_manager)
            answer_extractor = AnswerExtractor(string_cleaner)
        
            def extract_gt(answer, gt):
                try:
                    if gt != "" and not pd.isna(gt):
                        return gt
                    else:
                        if pd.isna(answer) or answer == "":
                            return None
                        else:
                            return answer_extractor.extract_answer(answer,None,True)
                except Exception as e:
                    self.logger.error(f"Error in extract_gt: {e}", exc_info=True)
                    return None
            
            # 使用 apply 遍历 DataFrame, 避免显式循环索引问题
            df[self.input_gt_key] = df.apply(lambda row: extract_gt(row[self.input_answer_key],
                                                                    row[self.input_gt_key]),
                                            axis=1)
        
        # 拆分有gt和无gt的 DataFrame
        df_with_gt = df[(df[self.input_gt_key].notna()) & (df[self.input_gt_key] != "")]
        df_without_gt = df[(df[self.input_gt_key].isna()) | (df[self.input_gt_key] == "")].copy()
        df_without_gt[self.input_gt_key] = None

        # 输出结果
        if len(df_with_gt) > 0:
            output_file_gt = storage.write(df_with_gt)
            self.logger.info(f"output {df_with_gt.shape[0]} rows with gt to {output_file_gt}")

        if len(df_without_gt) > 0:
            output_file_without_gt = storage.write(df_without_gt)
            self.logger.info(f"output {df_without_gt.shape[0]} rows without gt to {output_file_without_gt}")
                    
            



        
        