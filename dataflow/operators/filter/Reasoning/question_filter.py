from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC

import re

@OPERATOR_REGISTRY.register()
class QuestionFilter(OperatorABC):
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant.",
                 llm_serving: LLMServingABC = None,
                 prompt_template = None,
                 ):

        self.logger = get_logger()
        
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_serving = llm_serving
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对问题进行正确性检查，包括格式是否规范、语义是否合理、条件是否矛盾以及是否具备充分信息可解。"\n\n"
                "调用大语言模型依次执行四阶段判断，最终返回每个问题是否合格的二分类结果（保留合格样本）。\n\n"
                "输入参数：\n"
                "- system_prompt：系统提示词，用于定义模型行为\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- prompt_template：提示模板对象，用于构建检查提示词\n"
                "- input_key：输入问题字段名，默认为'math_problem'\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留判断结果为True的行\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator checks the correctness of questions, including formatting, semantic validity, logical consistency, "\n                "and whether the problem is solvable. It performs a four-stage evaluation using a large language model and retains qualified samples.\n\n"
                "Input Parameters:\n"
                "- system_prompt: System prompt to define model behavior\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- prompt_template: Prompt template object for constructing check prompts\n"
                "- input_key: Field name for input questions, default is 'math_problem'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only rows with True judgment results\n"
                "- List containing input field name for subsequent operator reference"
            )
        else:
            return (
                "QuestionFilter performs correctness checking on questions using a multi-stage LLM evaluation and retains qualified samples."
            )
    
    def ResolveResponse(self, response):
        try:
            pattern = re.compile(r'"judgement_test"\s*:\s*(true|false)', re.IGNORECASE)
            match = pattern.search(response)
            test_value = None
            if match:
                test_value = match.group(1).lower()
            else:
                if "true" in response.lower():
                    test_value = "true"
                else:
                    test_value = "false"
            if test_value == "true":
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Response format error for problem: {response}. Error: {e}")
            return False
            
    def run(self, storage: DataFlowStorage, input_key: str = "math_problem") -> list:
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        questions = dataframe[input_key]
        inputs = [self.prompt_template.build_prompt(question) for question in questions]
        responses = self.llm_serving.generate_from_input(user_inputs=inputs, system_prompt=self.system_prompt)
        results = [self.ResolveResponse(response) for response in responses]
        
        # 保留results为True的行
        dataframe = dataframe[results]
        output_file = storage.write(dataframe)
        self.logger.info(f"Filtered questions saved to {output_file}")
        
        return [input_key,]