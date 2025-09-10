from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import MetaPrompt  
import ast

example_dimensions = [
    {
        "dimension_name": "Text Structure",
        "description": "Evaluate the surface-level quality of the text, including spelling accuracy, grammar, vocabulary richness, and sentence structure.",
        "example_list": [
            {
                "text": "The experimental procedure was meticulously documented, with each variable clearly defined.",
                "score": "5"
            },
            {
                "text": "teh data was wrong and we dont no why it happen like that",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Diversity and Complexity",
        "description": "Assess how rich and conceptually varied the content is, and whether it requires expert or deep reasoning to understand.",
        "example_list": [
            {
                "text": "This article compares Bayesian inference and frequentist approaches in statistical modeling, highlighting theoretical and practical trade-offs.",
                "score": "5"
            },
            {
                "text": "Dogs are pets. They bark. They are friendly.",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Fluency and Understandability",
        "description": "Evaluate whether the text flows naturally, is easy to follow, and avoids awkward or disjointed phrasing.",
        "example_list": [
            {
                "text": "Despite initial challenges, the team successfully completed the deployment by adhering to a revised strategy.",
                "score": "5"
            },
            {
                "text": "The problem was and then fixed by something happens deployment successful maybe.",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Safety",
        "description": "Identify whether the text contains profanities, hate speech, or excessive personally identifiable information (PII).",
        "example_list": [
            {
                "text": "The software collects anonymous usage data to improve performance.",
                "score": "5"
            },
            {
                "text": "You idiot, your address 123 Main St will be posted online.",
                "score": "1"
            }
        ]
    },
    {
        "dimension_name": "Educational Value",
        "description": "Determine whether the text provides insight, stimulates thinking, or offers meaningful learning potential.",
        "example_list": [
            {
                "text": "Understanding the principles of thermodynamics allows engineers to design more efficient engines.",
                "score": "5"
            },
            {
                "text": "The sky is blue. Water is wet. This is how it is.",
                "score": "2"
            }
        ]
    },
    {
        "dimension_name": "Content Accuracy and Effectiveness",
        "description": "Assess the truthfulness, relevance, and practical usefulness of the content.",
        "example_list": [
            {
                "text": "Newton's second law states that F = ma, which explains the relationship between force, mass, and acceleration.",
                "score": "5"
            },
            {
                "text": "The Earth is flat and doesn't rotate around the Sun.",
                "score": "1"
            }
        ]
    }
]

@OPERATOR_REGISTRY.register()
class MetaSampleEvaluator(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 dimensions: list[dict] = example_dimensions,
                ):
        
        """
        Operator that evaluate the quality of the text based on the given dimensions.
        Argument Dimensions should be list of dict, each dict should contain:
        {
            "dimension_name": "Dimension Name",
            "description": "Description of the dimension",
            "example_list": [ // a list of example text and score
                {
                    "text": "example1 text to be evaluated",
                    "score": "the score of this dimension of the text above"
                },
                {
                    "text": "example2 text to be evaluated",
                    "score": "the score of this dimension of the text above"
                }            
            ]
        }
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'MetaScore'
        self.prompt = MetaPrompt(dimensions=dimensions)
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.dimensions = dimensions
        for item in dimensions:
            if 'dimension_name' not in item or 'description' not in item or 'example_list' not in item:
                raise ValueError('Invalid dimension format. Refer to the docstring for the correct format.')
            for example in item['example_list']:
                if 'text' not in example or 'score' not in example:
                    raise ValueError('Invalid example format. Refer to the docstring for the correct format.')
        self.output_columns = [item['dimension_name'] for item in dimensions]

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通过LLM评估文本的多个元属性，包括文本结构、多样性与复杂性、流畅性与可理解性、安全性、教育价值以及内容准确性与有效性。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- dimensions：评估维度列表，每个维度对应的字典中包含dimension_name，description，和示例字段：\n"
                "   * dimension_name：维度名称\n"
                "   * description：维度的描述\n"
                "   * example_list：包含示例文本和得分的列表\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含6个评估维度得分的DataFrame，列名为：Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, Content Accuracy & Effectiveness"
            )
        elif lang == "en":
            return (
                "Evaluate multiple meta attributes of text using LLM, including Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, and Content Accuracy & Effectiveness.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- dimensions: List of evaluation dimensions, each dimension corresponding to a dictionary containing dimension_name, description, and example field:\n"
                "   * dimension_name: Name of the dimension\n"
                "   * description: Description of the dimension\n"
                "   * example_list: List containing example texts and scores\n"
                "- input_key: Field name for input text\n"
                "Output Parameters:\n"
                "- DataFrame containing scores for 6 evaluation dimensions with columns: Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, Content Accuracy & Effectiveness"
            )
        else:
            return "Evaluate multiple meta attributes of text using LLM."
    
    def get_score(self, samples, input_key):
        system_prompt = self.prompt.build_system_prompt()
        user_prompts = []
        for sample in samples:
            input_text = sample.get(input_key, '')
            user_prompt = self.prompt.build_user_prompt(input_text)
            full_prompt = system_prompt + "\n" + user_prompt
            user_prompts.append(full_prompt)

        responses = self.llm_serving.generate_from_input(user_inputs=user_prompts)
        scores = []

        for i, response in enumerate(responses):
            try:
                lines = response.strip().split("\n")
                last_line = lines[-1].strip()
                parsed_scores = ast.literal_eval(last_line)
                if isinstance(parsed_scores, list) and len(parsed_scores) == 6:
                    scores.append(parsed_scores)
                else:
                    raise ValueError("Score format invalid")
            except Exception as e:
                self.logger.warning(f"Failed to extract score from response {i}: {e}")
                scores.append([float('nan')] * 6)

        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        samples = dataframe.to_dict(orient='records')
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = self.get_score(samples, input_key)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_key)
        # 展开6列固定命名
        score_df = pd.DataFrame(scores, columns=self.output_columns)
        dataframe = pd.concat([dataframe, score_df], axis=1)
        storage.write(dataframe)
