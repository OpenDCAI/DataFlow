from dataflow.operators.reasoning import (
    ReasoningQuestionFusionGenerator,
    ReasoningQuestionSolvableSampleEvaluator,
)
from dataflow.operators.core_text import (
    PandasOperator,
    EmbeddingGenerator,
)

from dataflow.prompts.reasoning.math import (
    MathQuestionParallelFusionGeneratorPrompt,
    MathQuestionSequentialFusionGeneratorPrompt,
    MathQuestionConditionFusionGeneratorPrompt,
    MathQuestionEvaluatorPrompt
)

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.core import LLMServingABC
import torch
import numpy as np
import pandas as pd
import re

# 这里或许未来可以有个pipeline基类
class ReasoningMath_APIPipeline_Mathfusion():
    def __init__(self, llm_serving: LLMServingABC = None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                    api_url="http://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=10
        )

        embedding_serving = APILLMServing_request(
                    api_url="http://api.openai.com/v1/embeddings",
                    model_name="text-embedding-ada-002",
                    max_workers=10
        )       
        
        self.embedding_generator = EmbeddingGenerator(embedding_serving=embedding_serving)
        
        def find_most_similar_questions(df):
            df = df.dropna(subset=['embeddings']).reset_index(drop=True)
            embeddings = torch.tensor(np.stack(df['embeddings'].values), dtype=torch.float32).cuda()  # shape: (n, d)
            sim_matrix = torch.matmul(embeddings, embeddings.T)  # shape: (n, n)
            sim_matrix.fill_diagonal_(-float('inf'))
            most_similar_idx = torch.argmax(sim_matrix, dim=1).cpu().numpy()
            df['most_similar_problem'] = df['question'].iloc[most_similar_idx].values

            return df
        
        self.most_similar_matcher = PandasOperator([
                        find_most_similar_questions
                    ])
        
        self.extract_pair = PandasOperator([ # dropping embeddings to decrease file size
                        lambda df: df.drop(columns=[col for col in df.columns if "embeddings" in col])
                    ])
        
        self.sequential_fusion = ReasoningQuestionFusionGenerator(
            num_prompts=1, 
            llm_serving=llm_serving, 
            prompt_template=MathQuestionSequentialFusionGeneratorPrompt()
            )

        self.parallel_fusion = ReasoningQuestionFusionGenerator(
            num_prompts=1, 
            llm_serving=llm_serving, 
            prompt_template=MathQuestionParallelFusionGeneratorPrompt()
            )

        self.condition_fusion = ReasoningQuestionFusionGenerator(
            num_prompts=2, 
            llm_serving=llm_serving, 
            prompt_template=MathQuestionConditionFusionGeneratorPrompt()
            )

        def combined(df: pd.DataFrame) -> pd.DataFrame:
            """
            Combine all question-related columns into a single long-format DataFrame.
            Automatically detects columns containing '_question_{i}' patterns.
            """
            # 始终保留原始 question
            question_cols = ["question"] if "question" in df.columns else []

            # 匹配所有类似 *_question_0, *_question_1, ...
            pattern = re.compile(r".*_question_\d+$")
            question_cols.extend([col for col in df.columns if pattern.match(col)])

            if not question_cols:
                raise ValueError("No question columns found matching pattern '_question_{i}'.")

            # 转换成长表
            long_df = df.melt(value_vars=question_cols, value_name="questions")[["questions"]]

            # 去除空值与重复项（通常同一问题会出现重复）
            long_df = long_df.dropna(subset=["questions"]).drop_duplicates().reset_index(drop=True)

            return long_df

        self.combined_question = PandasOperator([combined])

        self.question_evaluation = ReasoningQuestionSolvableSampleEvaluator(llm_serving=llm_serving, prompt_template=MathQuestionEvaluatorPrompt())

        def extract_new_problem(df: pd.DataFrame) -> pd.DataFrame:
            """
            Extract the content after '#New Problem#' from the 'questions' column
            and store it in a new column 'new_problem'.
            """
            if "questions" not in df.columns:
                raise ValueError("Input DataFrame must contain a 'questions' column.")

            def _extract(text: str) -> str:
                if not isinstance(text, str):
                    return None
                match = re.search(r"#New Problem#[:\s]*(.*)", text, re.DOTALL)
                return match.group(1).strip() if match else None

            df = df.copy()
            df["refined_question"] = df["questions"].apply(_extract)
            df = df.dropna(subset=["refined_question"]).reset_index(drop=True)

            return df
        
        self.extract_new_problem = PandasOperator([extract_new_problem])

    def forward(self):
        # self.first10.run(
        #     storage = self.storage.step(),
        # )

        self.embedding_generator.run(
            storage = self.storage.step(),
            input_key = "question",
            output_key = "embeddings",
        )

        self.most_similar_matcher.run(
            storage = self.storage.step(),
        )

        self.extract_pair.run(
            storage = self.storage.step(),
        )

        self.sequential_fusion.run(
            storage = self.storage.step(),
            input_problem_1= "question",
            input_problem_2= "most_similar_problem",
            output_key="sequential_fusion",
        )

        self.parallel_fusion.run(
            storage = self.storage.step(),
            input_problem_1= "question",
            input_problem_2= "most_similar_problem",
            output_key="parallel_fusion"
        )

        self.condition_fusion.run(
            storage = self.storage.step(),
            input_problem_1= "question",
            input_problem_2= "most_similar_problem",
            output_key="condition_fusion"
        )

        self.combined_question.run(
            storage = self.storage.step()
        )

        self.question_evaluation.run(
            storage = self.storage.step(),
            input_key = "questions",
            output_key= "question_solvability"
        )
        self.extract_new_problem.run(
            storage = self.storage.step()
        )



if __name__ == "__main__":
    pl = ReasoningMath_APIPipeline_Mathfusion()
    pl.forward()
