# 根节点，用来将数据拆入不同的分支
import pandas as pd
# from dataflow.generator.algorithms.AnswerExtraction_qwenmatheval import UnitTextManager,StringCleaner,AnswerExtractor
from dataflow.utils.utils import get_generator
import logging
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class AnswerPipelineRoot:
    def __init__(self, config: dict):
        self.config = config
        self.input_file = config.get("input_file")
        self.input_answer_key = config.get("input_answer_key")
        self.input_gt_key = config.get("input_gt_key")
        self.output_file_with_gt = config.get("output_file_with_gt")
        self.output_file_without_gt = config.get("output_file_without_gt")

    def run(self):
        # 读取输入文件
        df = pd.read_json(self.input_file, lines=True)

        if self.input_answer_key not in df.columns:
            raise ValueError(f"input_answer_key {self.input_answer_key} not in columns : {df.columns}")

        if self.input_gt_key not in df.columns or self.input_gt_key == "" or self.input_gt_key == None:
            # 直接copy input file 到 output file without gt
            df.to_json(self.output_file_without_gt, orient="records", lines=True)
            logging.warning(f"No gt in input file, copy input file to output file without gt")
            return


        # 初始化答案提取器
        unit_text_manager = get_generator("UnitTextManager")
        string_cleaner = get_generator("StringCleaner",unit_text_manager)
        answer_extractor = get_generator("AnswerExtractor",string_cleaner)

        # 如果不存在gt，则用answer传入answer_extractor.extract_answer()提取gt,否则直接使用gt
        def extract_gt(answer,gt):
            try:
                if gt != "" and not pd.isna(gt):
                    return gt
                else:
                    if pd.isna(answer):
                        return None
                    elif answer == "":
                        return None
                    else:
                        return answer_extractor.extract_answer(answer)
            except Exception as e:
                return None
            
        for _ in range(df.shape[0]):
            answer = df.loc[_,self.input_answer_key]
            gt = df.loc[_,self.input_gt_key]
            extracted_gt = extract_gt(answer,gt)
            
            df.loc[_,self.input_gt_key] = extracted_gt
        
        # 按有没有gt，将df拆成两个
        df_with_gt = df[df[self.input_gt_key].notna()]
        df_without_gt = df[df[self.input_gt_key].isna()]
        
        # 输出
        df_with_gt.to_json(self.output_file_with_gt, orient="records", lines=True)
        df_without_gt.to_json(self.output_file_without_gt, orient="records", lines=True)
        logging.info(f"output {df_with_gt.shape[0]} rows with gt to {self.output_file_with_gt}")
        logging.info(f"output {df_without_gt.shape[0]} rows without gt to {self.output_file_without_gt}")
                
        



        
        