from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import pandas as pd
import random
from dataflow.prompts.kbcleaning import MathVQAExtractPrompt
import os
from typing import List

@OPERATOR_REGISTRY.register()
class MathVQAExtractPicExtractor(OperatorABC):
    def __init__(self,
                llm_serving: LLMServingABC = None,
                model: str = "o4-mini",
                subject: str = "math"
                ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt = MathVQAExtractPrompt()
        self.model = model
        self.subject = subject

    def _format_instructions(self, image_files: List[str]):
        list_of_image_paths = []
        list_of_image_labels = []
        labels = ["page_" + image_file.split("_")[-1].split(".")[0] for image_file in image_files]
        for index in range(len(image_files)):
            list_of_image_paths.append([image_files[index]])
            list_of_image_labels.append([labels[index]])
        return list_of_image_paths, list_of_image_labels


    def run(self, storage, input_layout_path: str, output_folder: str):
        # 从layout_path/images中读取所有图片的文件名,确保为绝对路径
        image_files = [os.path.join(input_layout_path, image_file) for image_file in os.listdir(input_layout_path)]
        # 确保end with jpg & png
        image_files = [image_file for image_file in image_files if image_file.endswith(".jpg") or image_file.endswith(".png")]

        def filename2idx(filename: str):
            return int(filename.split("/")[-1].split(".")[0].split("_")[-1])
        # 按照文件名从小到大排序
        image_files.sort(key=filename2idx)

        list_of_image_paths, list_of_image_labels = self._format_instructions(image_files)
        system_prompt = self.prompt.build_prompt(self.subject)

        responses = self.llm_serving.generate_from_input_multi_images(list_of_image_paths, list_of_image_labels, system_prompt, self.model)

        # 将list of image paths和list of image labels和repsonses作为三列组织为jsonl
        list_of_dict = []
        for page, (image_path, image_label, response) in enumerate(zip(list_of_image_paths, list_of_image_labels, responses)):
            list_of_dict.append({"page": page, "image_path": image_path, "image_label": image_label, "response": response})
        df = pd.DataFrame(list_of_dict)

        # 将df保存为jsonl文件
        os.makedirs(output_folder, exist_ok=True)
        df.to_json(os.path.join(output_folder, "vqa_extract.jsonl"), orient="records", lines=True, force_ascii=False)

        return df
