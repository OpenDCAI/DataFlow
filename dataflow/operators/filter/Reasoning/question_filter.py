from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.reasoning import QuestionFilterPrompt
from dataflow.core import LLMServingABC

import re

@OPERATOR_REGISTRY.register()
class QuestionFilter(OperatorABC):
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant.",
                 llm_serving: LLMServingABC = None,
                 ):

        # self.check_config(config)
        self.logger = get_logger()
        self.system_prompt = system_prompt
        self.llm_serving = llm_serving


    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对数学问题进行正确性检查，包括格式是否规范、语义是否合理、条件是否矛盾以及是否具备充分信息可解。"
                "调用大语言模型依次执行四阶段判断，最终返回每个问题是否合格的二分类结果（0或1）。\n\n"
                "输入参数：\n"
                "- input_question_key：输入问题字段名\n"
                "- api_key：调用大模型所需的API密钥\n"
                "- model_name：使用的大模型名称\n"
                "- max_worker：并发线程数，用于加速处理\n\n"
                "输出参数：\n"
                "- result_key：判断结果字段名，值为0或1"
            )
        elif lang == "en":
            return (
                "This operator checks the correctness of math questions, including formatting, semantic validity, logical consistency, "
                "and whether the problem is solvable. It performs a four-stage evaluation using a large language model and returns a binary result (0 or 1).\n\n"
                "Input Parameters:\n"
                "- input_question_key: Field name for the input question\n"
                "- api_key: API key for calling the LLM\n"
                "- model_name: Name of the model used\n"
                "- max_worker: Number of threads for parallel processing\n\n"
                "Output Parameters:\n"
                "- result_key: Field name for the binary result, value is 0 or 1"
            )
        else:
            return (
                "QuestionFilter performs correctness checking on math questions using a multi-stage LLM evaluation and returns binary results (0/1)."
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
        inputs = [QuestionFilterPrompt().build_prompt(question) for question in questions]
        responses = self.llm_serving.generate_from_input(user_inputs=inputs, system_prompt=self.system_prompt)
        results = [self.ResolveResponse(response) for response in responses]
        
        # 保留results为True的行
        dataframe = dataframe[results]
        output_file = storage.write(dataframe)
        self.logger.info(f"Filtered questions saved to {output_file}")
        
        return [input_key,]