import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import json

import json
import re


@OPERATOR_REGISTRY.register()
class ExtractSmilesFromText(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC, prompt_template = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.json_failures = 0
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "ExtractSmilesFromText 用于从 OCR 文本中抽取或解析化学分子的 SMILES 表达式。"
                "算子会根据给定的提示模板（prompt_template），结合文本内容和（可选的）单体缩写信息，"
                "调用大语言模型完成解析与结构化，并将结果以 JSON 格式写回到指定列。\n"
                "\n输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- prompt_template：提示词模板对象，用于构造模型输入\n"
                "- content_key：包含 OCR 文本的列名（默认 'text'）\n"
                "- abbreviation_key：包含缩写/单体信息的列名（默认 'abbreviations'），可为空\n"
                "- output_key：写回抽取结果的列名（默认 'synth_smiles'）\n"
                "\n输出参数：\n"
                "- DataFrame，其中 output_key 列为模型返回并经 JSON 解析后的 SMILES 结构\n"
                "- 返回 output_key，供后续算子引用\n"
                "\n备注：\n"
                "- 模型输出会尝试解析为 JSON；若解析失败，将返回 [] 并记录失败次数。"
            )
        elif lang == "en":
            return (
                "ExtractSmilesFromText is designed to extract or parse chemical SMILES expressions "
                "from OCR text. It uses a given prompt_template to construct model inputs, combining "
                "text content and optional abbreviation/monomer information, then calls the LLM to "
                "produce structured outputs which are parsed into JSON.\n"
                "\nInput Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- prompt_template: Prompt template object used to build model input\n"
                "- content_key: Column name containing OCR text (default 'text')\n"
                "- abbreviation_key: Column name containing abbreviations/monomer info (default 'abbreviations'); optional\n"
                "- output_key: Column name to store extracted results (default 'synth_smiles')\n"
                "\nOutput:\n"
                "- DataFrame with output_key column containing JSON-parsed SMILES structures\n"
                "- Returns output_key for downstream operators\n"
                "\nNotes:\n"
                "- The operator attempts to parse model outputs as JSON; failures return [] and are counted."
            )
        else:
            return "ExtractSmilesFromText extracts chemical SMILES expressions from OCR text using an LLM."

    def _strip_code_fence(self, s: str) -> str:
        s = s.strip()
        # 去掉 ```json ... ``` 或 ``` ... ```
        if s.startswith("```"):
            # 去掉第一行的 ```(json)?
            s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
            # 去掉结尾 ```
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _safe_json_load(self, item):
        """
        尝试把 item 解析为 JSON：
        - 解析失败：返回 []，并将 self.json_failures += 1
        - 解析成功：返回解析后的对象（list/dict/其他 JSON 标准类型）
        """
        try:
            # 已经是结构化就直接返回
            if isinstance(item, (list, dict)):
                return item
            if item is None:
                return []  # 约定空返回

            # 其他类型（比如 float/int）转成字符串再处理
            if not isinstance(item, str):
                item = str(item)

            s = item.strip()
            if not s:
                return []  # 空字符串

            # 去掉代码块围栏
            s = self._strip_code_fence(s)

            # 去掉包裹的引号（例如整个内容被 "..." 或 '...' 包着）
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()

            # 去掉前缀 json/JSON
            s = re.sub(r"^\s*(?:json|JSON)\s*", "", s)

            # 定位第一个 { 或 [，从这里开始截
            m = re.search(r"[\[\{]", s)
            if m:
                s = s[m.start():].strip()

            # 如果末尾有多余字符，尝试保留到最后一个 ] 或 }
            last_bracket = max(s.rfind("]"), s.rfind("}"))
            if last_bracket != -1:
                s = s[:last_bracket + 1].strip()

            # 第一次解析
            obj = json.loads(s)

            # 如果第一次解析结果仍是字符串，尝试再解一次（处理二次编码）
            if isinstance(obj, str):
                try:
                    obj2 = json.loads(obj)
                    return obj2
                except json.JSONDecodeError:
                    # 二次解析失败不视为致命，返回第一次结果
                    return obj

            return obj

        except Exception as e:
            # 任何异常：计数 + 返回空列表
            self.json_failures += 1
            # 打印精简预览，避免日志过长
            preview = ""
            try:
                preview = (s if len(s) <= 200 else s[:200] + "...").replace("\n", "\\n")
            except Exception:
                preview = "<unavailable>"
            self.logger.warning(f"[safe_json_load] 解析失败，第{self.json_failures}次；错误：{type(e).__name__}: {e}；预览: {preview}")
            return []

    def run(self, storage: DataFlowStorage, content_key: str = "text", abbreviation_key: str = "abbreviations", output_key: str = "synth_smiles"):
        # self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        llm_inputs = []

        # Prepare LLM inputs by formatting the prompt with raw content from the dataframe
        for index, row in dataframe.iterrows():
            content = row.get(content_key, '')
            monomer = row.get(abbreviation_key, '')
            llm_input = self.prompt_template.build_prompt(monomer) + content 
            llm_inputs.append(llm_input)
        
        # Generate the text using the model
        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return

        parsed_outputs = []
        for item in generated_outputs:
            try:
                parsed_outputs.append(self._safe_json_load(item))
            except Exception:
                parsed_outputs.append([])
        #parsed_outputs = [self._safe_json_load(item)['chemical_structures'] for item in generated_outputs]
        # print(parsed_outputs)
        dataframe[output_key] = parsed_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key

