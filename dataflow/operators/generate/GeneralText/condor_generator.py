import json
import random
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import CondorPrompt


@OPERATOR_REGISTRY.register()
class CondorGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, num_samples=15):
        # Based on the existing topics, it is recommended to set num_samples below 5000. Otherwise, it is recommended to add topics in dataflow.prompts.general_text.CondorPrompt on your own to increase data richness
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.num_questions = num_samples // 3
        self.prompt = CondorPrompt() 
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于预置知识树标签，两阶段从0合成SFT格式数据（合成数量大于5000时建议增加标签数量）。第一阶段生成不同难度级别的问题，第二阶段为每个问题生成对应的答案。"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- num_samples：生成样本总数，建议小于5000，默认值为15\n"
                "输出参数：\n"
                "- 包含'difficulty'、'instruction'和'output'字段的DataFrame\n"
                "- 返回生成的DataFrame用于后续处理"
            )
        elif lang == "en":
            return (
                "Two-stage generation of SFT-style data from scratch based on predefined knowledge tree tags (for over 5000 samples, consider increasing the number of tags). \n"
                "First stage generates questions of varying difficulty levels, second stage generates answers for each question.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- num_samples: Total number of samples to generate, recommended to be less than 5000, default is 15\n\n"
                "Output Parameters:\n"
                "- DataFrame containing 'difficulty', 'instruction', and 'output' fields\n"
                "- Returns generated DataFrame for subsequent processing"
            )
        else:
            return (
                "CondorGenerator generates SFT-style data through two-stage LLM generation based on predefined knowledge tree tags."
            )

    
    def parse_generated_responses(self, questions_responses):
        questions_data = []
        for response in questions_responses:
            try:
                if not isinstance(response, str):
                    raise ValueError("Invalid response type")
                
                # 解析问题字符串，提取不同难度级别的问题
                question_data = {}
                lines = response.split('\n')

                for line in lines:
                    if line.startswith("[Easy]"):
                        question_data["Easy"] = line.replace("[Easy][Question Start]", "").replace("[Question End]", "").strip()
                    elif line.startswith("[Medium]"):
                        question_data["Medium"] = line.replace("[Medium][Question Start]", "").replace("[Question End]", "").strip()
                    elif line.startswith("[Hard]"):
                        question_data["Hard"] = line.replace("[Hard][Question Start]", "").replace("[Question End]", "").strip()

                if question_data:
                    questions_data.append(question_data)
            except Exception as e:
                self.logger.debug(f'Error while parsing the response: {str(e)}')
                continue

        return questions_data

    def run(self, storage: DataFlowStorage):
        # 生成所有的prompt
        prompts = []
        for _ in range(self.num_questions):
            # 每次随机选择topic, domain, theme
            topic = random.choice(list(self.prompt.tag.keys()))
            domain = random.choice(list(self.prompt.tag[topic].keys()))
            theme = random.choice(self.prompt.tag[topic][domain])
            # 获取生成问题的prompt
            prompt = self.prompt.get_question_prompt(theme, domain)
            prompts.append(prompt)

        
        # 调用LLM一次性生成问题
        self.logger.info("Generating questions...")
        questions_responses = self.llm_serving.generate_from_input(user_inputs=prompts)

        # 解析问题
        self.logger.info("Parsing questions...")
        questions_data = self.parse_generated_responses(questions_responses)
        # 生成答案的prompt
        answer_prompts = []
        for question in questions_data:
            for difficulty_level in ["Easy", "Medium", "Hard"]:
                question_text = question.get(difficulty_level)
                if question_text:
                    # 构造问题对应的answer prompt
                    answer_prompt = f"Please answer this questiong truthfully. Question: {question_text}"
                    answer_prompts.append(answer_prompt)

        # 调用LLM一次性生成所有答案
        self.logger.info("Generating answers...")
        answer_responses = self.llm_serving.generate_from_input(user_inputs=answer_prompts)

        # 解析答案
        answers_data = []
        answer_idx = 0  # 用来追踪答案响应的索引
        for question in questions_data:
            for difficulty_level in ["Easy", "Medium", "Hard"]:
                question_text = question.get(difficulty_level)
                if question_text:
                    # 获取对应的答案
                    answer_text = answer_responses[answer_idx].strip()  # 获取答案
                    answers_data.append({
                        "difficulty": difficulty_level,
                        "instruction": question_text,
                        "output": answer_text
                    })
                    answer_idx += 1  # 更新索引

        # Step 4: Write to storage (e.g., save to DataFrame)
        df = pd.DataFrame(answers_data)
        storage.write(df)
        self.logger.info(f'SFT data generated')
        return df
