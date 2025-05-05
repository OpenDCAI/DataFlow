import json
import logging
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import re
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request

def extract_json_objects(model_output, nums=3):
    """
    从大模型输出中提取nums个JSON对象
    假设模型的输出包含了nums个按 JSON 格式排列的问题和答案
    """
    # 定义匹配 JSON 格式的正则表达式
    json_pattern = r'\{[^}]*\}'

    # 查找所有符合格式的 JSON 字符串
    matches = re.findall(json_pattern, model_output)

    # if len(matches) != nums:
    #     raise ValueError(f"Expected exactly {nums} JSON objects in the output.")

    # 解析匹配到的 JSON 对象
    json_entries = []
    for match in matches:
        try:
            json_obj = json.loads(match)
            # 确保每个 JSON 对象都有 "instruction" 和 "output" 字段
            if 'instruction' in json_obj and 'output' in json_obj:
                json_entries.append(json_obj)
            else:
                raise ValueError("JSON object is missing 'instruction' or 'output'.")
        except json.JSONDecodeError:
            continue

    return json_entries

@GENERATOR_REGISTRY.register()
class SupervisedFinetuneGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.input_file = config['input_file']
        self.output_file = config['output_file']
        self.key = config['keys']
        self.model = self.__init_model__()

    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()

        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")

    def run(self):
        raw_dataframe = pd.read_json(self.input_file, lines=True)
        
        num_questions = 1
        prompt = f"""You are tasked with creating high-quality SFT data for LLM training. Please generate a batch of data where the only task is to create questions based on the provided context. Focus on diversity, relevance, and clarity when forming the questions. Please ensure each question is distinct and appropriately aligned with the context.
            Your Questions must meet the following requirements:
            1. You must strictly create exactly {num_questions} questions at a time.
            2. These {num_questions} questions must cover different aspects of the topic, ensuring diversity and comprehensiveness.
            3. Each question should be framed based on the provided context and should include sufficient background information. The questions should not be abrupt or vague.
            4. Please output the {num_questions} questions in the following JSON format:"""
            
        # 动态构建第四点中的JSON格式示例部分
        example_questions = []
        for i in range(1, num_questions + 1):
            example_questions.append(f"""
                {{
                    "instruction": "QUESTION{i}",
                    "output": "ANSWER{i}"
                }}""")

        # 将所有生成的示例问题部分拼接在一起
        example_questions_str = ",".join(example_questions)

        # 添加示例问题到prompt
        prompt += example_questions_str

        # 添加示例数据
        prompt += f"""
            5. Here are some examples of high-quality SFT data questions with completed outputs:
            {{
                "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points.",
                "output": "Here's an HTML page with bullet points for healthy habits: <html><body><h3>Healthy Habits:</h3><ul><li>Eating a balanced diet with plenty of fruits and vegetables.</li><li>Engaging in regular physical activity, such as walking, running, or cycling.</li><li>Getting enough sleep each night, ideally 7-8 hours.</li><li>Staying hydrated by drinking plenty of water throughout the day.</li><li>Limiting alcohol consumption and avoiding smoking.</li><li>Managing stress through relaxation techniques like meditation or yoga.</li><li>Regularly visiting a healthcare provider for check-ups and preventative care.</li></ul></body></html>"
            }},
            {{
                "instruction": "How can we use Python to calculate the GCD (greatest common divisor) of five numbers and express each number in terms of the GCD?",
                "output": "Here's a Python function that calculates the GCD of five numbers: def find_gcd(a, b, c, d, e): smallest = min(a, b, c, d, e) while True: if all(x % smallest == 0 for x in [a, b, c, d, e]): return smallest else: smallest -= 1. This will compute the GCD and can express each number in terms of the GCD."
            }},
            {{
                "instruction": "Generate a list of ten essential items a person might need for a camping trip, considering the weather, terrain, and local wildlife. Provide a breakdown of weight, size, and functionality of each item.",
                "output": "Here's a list of ten essential items for a camping trip: 1. Tent - lightweight, easy to set up, weighs 3.5 pounds, $450. 2. Sleeping bag - keeps you warm at 20°F, weighs 2 pounds, $200. 3. Sleeping pad - insulates and cushions, weighs 1 pound, $100. 4. Backpack - 65L capacity, weighs 4.5 pounds, $250. 5. Water filter - lightweight, filters up to 100,000 gallons, $40. 6. Headlamp - bright, 300 lumens, $30. 7. Multi-tool - versatile, 18 tools, $80. 8. Stove - boils water in 2 minutes, $100. 9. Bear canister - stores food securely, $70. 10. First aid kit - essential medical supplies, $50."
            }}
            6. Now it's your turn. You can use your rich imagination, but note that you cannot copy the expression from the examples; you must have your own new expression:

            Please create {num_questions} distinct and well-formed questions based on the following context:"""

        print(raw_dataframe.columns.to_list())
        raw_dataframe_copy = raw_dataframe.copy()
        print(raw_dataframe.columns.to_list())
                
        llm_inputs = []
        inputs = []
        for input in raw_dataframe[self.key]:
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': input},
            ]
            def messages_to_prompt(messages):
                prompt = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                prompt += "<|im_start|>assistant"
                return prompt
            llm_inputs.append(messages_to_prompt(messages))
        
        outputs = self.model.generate_text_from_input(llm_inputs)
        
        raw_dataframe['output'] = outputs
        
        qa_list = [row['output'] for _, row in raw_dataframe.iterrows()]
        qa_list_new = []
        
        for output in outputs:
            # 提取JSON对象
            json_objects = extract_json_objects(output, nums=num_questions)
            qa_list_new.append(json_objects)


        raw_dataframe = raw_dataframe_copy.loc[raw_dataframe_copy.index].reset_index(drop=True)
        raw_dataframe['generated_content'] = qa_list_new

        # 初始化一个新的列表，用于存储处理后的行
        expanded_rows = []

        # 遍历原始数据中的每一行
        for index, row in raw_dataframe.iterrows():
            # 获取generated_content的个数
            num_generated = len(row['generated_content'])
            generated_content = row['generated_content']
            # 根据generated_content的个数，生成对应数量的行
            for i in range(num_generated):
                expanded_row = row.copy()  # 复制原始行
                expanded_row['instruction'] = generated_content[i]['instruction']
                expanded_row['output'] = generated_content[i]['output']
                # 删除generated_content列
                if 'generated_content' in expanded_row:
                    del expanded_row['generated_content']
                expanded_rows.append(expanded_row)


        # 用新的行列表更新原始DataFrame
        raw_dataframe = pd.DataFrame(expanded_rows)

        raw_dataframe.to_json(self.output_file, orient='records', lines=True)   
