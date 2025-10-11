
"""
本文件用于从零合成SFT训练数据，主要流程包括：
1. 通过PromptedGenerator生成原始数据（raw_generation）；
2. 对原始数据进行重写（rewrite），丰富多样性；
3. 对原始和重写数据分别打分（score_raw, score_rewrite）；
4. 过滤低分样本，仅保留高质量数据；
5. 最终整理为SFT训练所需的标准格式。

使用方法：
直接运行本文件即可自动完成上述流程，最终结果会保存在指定的cache目录下。
"""
from dataflow.operators.core_text import RandomDomainKnowledgeRowGenerator
from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.prompts.core_sft_from_scratch import *
import pandas as pd
from dataflow.prompts.core_sft_from_scratch import (
    SFTFromScratchGeneratorPrompt
)

def GetSFTFromScratchRewriterPrompt():
    system_prompt = """You are an expert data diversity specialist focused on generating varied, high-quality SFT training samples.

    Your task is to rewrite existing SFT samples to maximize training diversity while preserving their educational value and correctness.

    ## Rewriting Strategies:

    **Linguistic Variation:**
    - Alter sentence structure and word choice
    - Change formality levels (formal ↔ casual)
    - Vary instruction phrasing patterns
    - Use synonyms and alternative expressions

    **Perspective Shifts:**
    - Change user personas or contexts
    - Modify scenarios while keeping core requirements
    - Adjust complexity or specificity levels
    - Alter examples or use cases

    **Content Enhancement:**
    - Add relevant details or context where beneficial
    - Reorganize information for better clarity
    - Include alternative approaches or explanations
    - Expand or condense based on appropriateness

    ## Preservation Requirements:
    - Maintain factual accuracy and correctness
    - Keep the same domain classification
    - Preserve the core learning objective
    - Ensure output quality remains high or improves

    ## Quality Standards:
    - Natural, fluent language throughout
    - Logical flow between instruction, input, and output
    - Appropriate depth and comprehensiveness
    - Professional yet accessible tone

    Your rewritten version should feel like a completely different person asking a related question, while maintaining the same educational value."""
            
    user_prompt = f"""Transform the following SFT sample into a high-quality variant that maximizes diversity while preserving correctness and educational value.


    ## Rewriting Instructions:
    1. Significantly change the wording and structure
    2. Consider altering the context, scenario, or user perspective
    3. Maintain or improve the response quality and completeness
    4. Keep the same domain but feel free to adjust the specific focus within that domain
    5. Ensure the result sounds natural and represents authentic user needs

    Output Format: Single-line JSON with keys: instruction, input, output, domain
    No explanations - output only the JSON."""
    return system_prompt + "\n\n" + user_prompt

def GetSFTFromScratchScorerPrompt():
    system_prompt = """You are a meticulous SFT data quality evaluator with expertise in machine learning training data.

    Your role is to assess the overall quality of SFT training samples using a comprehensive evaluation framework.

    ## Scoring Scale (1-5):
    **5 - Excellent:** 
    - Crystal clear, well-structured instruction representing authentic user needs
    - Response is comprehensive, accurate, and demonstrates expertise
    - Perfect alignment between instruction and output
    - Natural language with appropriate depth and examples
    - Would significantly contribute to model training

    **4 - Good:**
    - Clear instruction with minor room for improvement
    - Solid, helpful response with good accuracy
    - Strong instruction-output alignment
    - Generally well-written with adequate detail
    - Valuable training contribution

    **3 - Acceptable:**
    - Understandable instruction, though potentially generic
    - Correct but basic response
    - Reasonable alignment between components
    - Functional but unremarkable quality
    - Standard training value

    **2 - Poor:**
    - Unclear, ambiguous, or poorly constructed instruction
    - Response has accuracy issues or lacks depth
    - Weak alignment between instruction and output
    - Unnatural language or inappropriate complexity
    - Limited training value

    **1 - Very Poor:**
    - Confusing or meaningless instruction
    - Incorrect, harmful, or severely inadequate response
    - No clear relationship between components
    - Major language or formatting issues
    - Detrimental to training quality

    ## Evaluation Criteria:
    - **Clarity & Specificity:** Is the instruction clear and actionable?
    - **Response Quality:** Is the output accurate, comprehensive, and helpful?
    - **Alignment:** Do instruction, input, and output work together logically?
    - **Authenticity:** Does this represent realistic user-AI interaction?
    - **Educational Value:** Would this improve model performance?
    - **Language Quality:** Is the text natural, well-written, and appropriate?

    REMEMBER: Output ONLY the integer score. Nothing else."""
            
    user_prompt = f"""Score this SFT training sample (1-5 integer only):
    Output format: single integer 1-5, no other text"""
    return system_prompt + "\n\n" + user_prompt


#example domain_seed_keys
domain_keys = {
    "general_conversation": "General conversation and casual chatting",
    "knowledge_qa": "Knowledge-based question answering (history, science, culture, etc.)",
    "summarization": "Summarization of text, paragraphs, or articles",
    "paraphrasing": "Rewriting text to keep the same meaning",
    "translation": "Translation between languages",
    "reasoning": "Logical reasoning, step-by-step thinking, and math problems",
    "creative_writing": "Poetry, stories, slogans, and creative tasks",
    "coding": "Programming tasks, code explanation, debugging",
    "instruction_explanation": "Explain how to do something step by step",
    "data_analysis": "Analyze and interpret data (tables, charts)",
    "academic_writing": "Essays, research summaries, and academic-style writing",
    "legal": "Legal Q&A, contract summarization, compliance checks",
    "medical": "Healthcare advice, symptoms explanation (non-diagnostic)",
    "finance": "Investment concepts, financial analysis, budgeting",
    "business": "Marketing, product descriptions, business strategy",
    "technology": "Tech explanations, software tutorials, AI concepts",
    "education": "Teaching concepts, lesson plans, educational Q&A"
}



class CoreSftFromScratchPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./dataflow_cache",  
            file_name_prefix="core_sft_from_scratch_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './core_sft_cache'
        self.generated_samples_num = 10

        llm_serving = APILLMServing_request(
            api_url="http://your-api-url/v1/chat/completions", 
            model_name="gpt-4o",
            max_workers=100
        )

        self.generator = RandomDomainKnowledgeRowGenerator(
            llm_serving=llm_serving,
            prompt_template=SFTFromScratchGeneratorPrompt()
        )
        self.rewriter = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetSFTFromScratchRewriterPrompt(),
        )
        self.scorer = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetSFTFromScratchScorerPrompt(),
        )
        self.filter = GeneralFilter([
            lambda df: df['score_raw'] > 3,
            lambda df: df['score_rewrite'] > 3
        ])

    def get_final_data(self):
        df = self.storage.step().read("dataframe")
        sft_rows = []
        for _, row in df.iterrows():
            for col in ["raw_generation", "rewrite"]:
                value = row.get(col, None)
                if pd.notnull(value) and str(value).strip():
                    try:
                        # 假设每条内容是单行JSON
                        sample = eval(value) if isinstance(value, str) else value
                        # 只保留格式正确的数据
                        if (
                            isinstance(sample, dict)
                            and all(k in sample for k in ["instruction", "input", "output", "domain"])
                        ):
                            sft_rows.append(sample)
                    except Exception:
                        continue
        sft_df = pd.DataFrame(sft_rows)
        self.storage.write(sft_df)

    def forward(self):
        # 生成
        self.generator.run(
            storage=self.storage.step(),
            generation_num=self.generated_samples_num // 2,
            output_key="raw_generation",
            domain_keys=str(domain_keys.keys())
        )
        # 重写，丰富
        self.rewriter.run(
            storage=self.storage.step(),
            input_key="raw_generation",
            output_key="rewrite"
        )
        # 给数据进行打分
        self.scorer.run(
            storage=self.storage.step(),
            input_key="raw_generation",
            output_key="score_raw"
        )
        self.scorer.run(
            storage=self.storage.step(),
            input_key="rewrite",
            output_key="score_rewrite"
        )
        # 过滤
        self.filter.run(
            storage=self.storage.step()
        )
        self.get_final_data()


if __name__ == "__main__":
    model = CoreSftFromScratchPipeline()
    model.forward()
