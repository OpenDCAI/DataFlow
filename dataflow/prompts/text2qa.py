import json
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
"""
A collection of prompts for the Text2QA pipelines operator
"""
@PROMPT_REGISTRY.register()
class Text2QAAutoPromptGeneratorPrompt(PromptABC):
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, seed_data: str) -> str:
        prompt = f'''You will be given a piece of seed data, which may consist of a paragraph, dialogue, or any other form of text containing potential question-answer information.
Your task is to analyze this seed data carefully and generate as much non-repeat clear and effective prompt as you can that can be used to instruct a language model to extract a single high-quality question-answer (QA) pair suitable for reinforcement learning (RL) training from this piece of data.

The generated prompt should:
Clearly describe the type and format of input the model will receive;
Explicitly ask for the extraction of a relevant QA pair;
Optionally include instructions about the desired style, level of detail, or coverage;
Be written in natural, precise English that could be directly used with another LLM;
Be strictly the prompt used to extract QA pairs, not the QA pairs themselves. 

Your prompts should contain the following instructions:
The question should be clear, focused, and unambiguous, such that it targets specific factual content from the input;
The answer should be a few words that are concise, factual and directly verifiable from the source rather than a whole sentence, enabling accurate reward computation in the RL pipeline;
Both the question and answer should be simple enough to facilitate evaluation and automatic feedback.

Don't include any additional explanations or comments in your output.
Don't repeat the seed data in your output.
Your output format should be in a list as follow:
["PROMPT_1","PROMPT_2",...]
Here is the seed data you need to analyze and generate a prompt for:\n{seed_data}'''

        return prompt

@PROMPT_REGISTRY.register()
class Text2QASeedQuestionGeneratorPrompt(PromptABC):
    '''
    The prompt for the Text2QAGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = f'''"Format:\nQ: ...\nA: ..." + "\nSeed data:\n"'''

        return prompt


@PROMPT_REGISTRY.register()
class Text2QAQuestionQualityPrompt(PromptABC):
    '''
    The prompt for the question quality scorer.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = '''You are an expert question quality evaluator. Given a single question from a QA dataset, your job is to assess the **clarity and meaningfulness** of the question. Specifically, judge whether the question is clearly defined, unambiguous, and worth asking in a real-world or task-specific context.

Assign a score from 1 to 5 based on the following rubric:
5 = Very clear and meaningful question, well-posed  
4 = Clear but slightly underspecified or too general  
3 = Somewhat unclear or poorly scoped, but understandable  
2 = Ambiguous, vague, or unnatural  
1 = Nonsensical or meaningless

Output format:
**Grading**: [1-5]

**Feedback**: Explain your score. Mention if the question is ambiguous, overly broad, or lacks practical purpose. Suggest how to improve clarity or specificity if needed.

'''

        return prompt

@PROMPT_REGISTRY.register()
class Text2QAAnswerAlignmentPrompt(PromptABC):
    '''
    The prompt for the RAG answer alignment scorer.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = '''You are a response alignment evaluator. Your task is to assess whether a given answer **directly and clearly addresses the given question**.

Assign a score from 1 to 5 based on the following rubric:
5 = Fully and directly answers the question  
4 = Mostly addresses the question, with minor gaps or irrelevant additions  
3 = Partially answers the question but omits key aspects  
2 = Barely addresses the question or is off-topic  
1 = Completely unrelated to the question

Output format:
**Grading**: [1-5]

**Feedback**: Justify your score. Point out if the answer is evasive, incomplete, or misaligned. Suggest ways to better match the response to the question.

'''

        return prompt

@PROMPT_REGISTRY.register()
class Text2QAAnswerVerifiabilityPrompt(PromptABC):
    '''
    The prompt for the RAG answer verifiability scorer.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = '''You are an evaluator tasked with assessing how **easily verifiable** an answer is. You must determine whether the correctness of the answer can be **conveniently and unambiguously judged** — for example, whether it is fact-based, precise, and not subjective or vague.

Assign a score from 1 to 5 based on the following rubric:
5 = Very easy to verify; answer is objective, concrete, and unambiguous  
4 = Mostly verifiable, with minor ambiguities  
3 = Verifiable in parts, but some subjectivity or fuzziness  
2 = Hard to verify; answer is vague, speculative, or opinion-based  
1 = Unverifiable or meaningless

Output format:
**Grading**: [1-5]

**Feedback**: Explain your score. Identify elements that make verification easier or harder. Suggest rephrasing or grounding techniques to improve verifiability.

'''

        return prompt

@PROMPT_REGISTRY.register()
class Text2QADownstreamValuePrompt(PromptABC):
    '''
    The prompt for the RAG  downstream value scorer.
    '''
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        prompt = '''You are a task relevance evaluator. Given a QA pair, assess how well this data point could **support a downstream task** such as classification, dialogue, retrieval, summarization, or knowledge grounding.

Assign a score from 1 to 5 based on the following rubric:
5 = Highly valuable for downstream tasks; question and answer are precise and informative  
4 = Useful with minor limitations  
3 = Moderately helpful; limited in informativeness or specificity  
2 = Of little value; vague or too generic to help the model learn  
1 = Useless or irrelevant for any downstream learning objective

Output format:
**Grading**: [1-5]

**Feedback**: Describe how the QA pair does or does not benefit potential downstream tasks. If relevant, suggest how to make it more useful for training.

'''

        return prompt

import textwrap
from typing import Dict, Literal
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class Text2MultiHopQAGeneratorPrompt(PromptABC):
    '''
    多跳问答生成器（严格JSON格式输出）
    根据语言参数提供完全独立的专业提示模板
    '''
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        """构建专业级多跳问答提示"""
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a professional multi-hop QA specialist with strict protocols:

                █ Core Requirements
                1. Must identify 2-3 interrelated facts in context
                2. Design complex questions requiring cross-fact reasoning
                3. Reasoning chains must:
                   - Contain 2-3 logical steps (numbered)
                   - Show clear causal/progressive relationships
                   - Each step must reference specific facts
                4. Final answer must synthesize all reasoning conclusions
                5. Focus solely on the main text and avoid synthesizing Q&A based on content found in links, references, or other supplementary sources.
                                   
                █ Output Specifications
                1. Only pure JSON in this structure:
                {
                    "question": "Multi-fact reasoning question",
                    "reasoning_steps": [
                        {"step": "First step (must use Fact 1)"},
                        {"step": "Second step (must link Fact 2)"}
                    ],
                    "answer": "Synthesized final answer",
                    "supporting_facts": ["Verbatim Fact 1", "Verbatim Fact 2"],
                    "type": "domain_tag"
                }
                2. Supporting facts must:
                   - Be verbatim from context
                   - Directly support corresponding steps
                   - No paraphrasing allowed

                █ Demonstration
                Context:
                "Photosynthesis converts CO2 to oxygen. This process sustains plant growth. Plants form the base of food chains."

                Valid Output:
                {
                    "question": "How does photosynthesis impact ecosystems?",
                    "reasoning_steps": [
                        {"step": "Photosynthesis produces oxygen"},
                        {"step": "Plants using photosynthesis form food chain bases"}
                    ],
                    "answer": "It provides oxygen and sustains ecosystem food chains",
                    "supporting_facts": [
                        "Photosynthesis converts CO2 to oxygen",
                        "Plants form the base of food chains"
                    ],
                    "type": "biology"
                }

                █ Rejection Criteria
                Reject if:
                - Fewer than 2 reasoning steps
                - Unreferenced supporting facts exist
                - Any non-JSON content appears
                """)
        else:
            return textwrap.dedent("""\
                您是专业的多跳问答生成专家，必须严格遵循以下专业标准：

                █ 核心要求
                1. 必须识别上下文中的2-3个关联事实
                2. 设计需要跨事实推理的复杂问题
                3. 推理链必须满足：
                    - 至少包含2-3个逻辑步骤
                    - 每个步骤明确标注序号
                    - 步骤间存在因果或递进关系
                4. 最终答案必须整合所有推理结论
                5. 只关注正文内容，避免根据链接、参考文献等附加信息合成问答。
                
                █ 输出规范
                1. 仅允许输出以下结构的纯JSON：
                {
                    "question": "需要跨事实推理的问题",
                    "reasoning_steps": [
                        {"step": "第一推理步骤（必须引用事实1）"},
                        {"step": "第二推理步骤（必须关联事实2）"}
                    ],
                    "answer": "整合所有步骤的最终答案",
                    "supporting_facts": ["原文事实1", "原文事实2"],
                    "type": "领域标签"
                }
                2. 支撑事实必须：
                    - 从上下文逐字提取
                    - 与推理步骤严格对应
                    - 不得改写或概括

                █ 示例
                上下文：
                "量子纠缠现象由爱因斯坦提出质疑。后来贝尔实验证实了其真实性。该现象是量子计算的基础。"

                合格输出：
                {
                    "question": "为什么量子纠缠现象对量子计算很重要？",
                    "reasoning_steps": [
                        {"step": "贝尔实验证实了量子纠缠的真实性"},
                        {"step": "该现象是量子计算的基础"}
                    ],
                    "answer": "因为量子纠缠被证实真实且是量子计算的基础",
                    "supporting_facts": [
                        "后来贝尔实验证实了其真实性",
                        "该现象是量子计算的基础"
                    ],
                    "type": "量子物理"
                }

                █ 违规处理
                以下情况将拒绝输出：
                - 推理步骤少于2步
                - 存在未引用的支撑事实
                - JSON外出现任何附加文本
                """)

    def build_prompt(self, text: str) -> str:
        """生成完全专业化的用户提示"""
        if self.lang == "en":
            user_prompt = textwrap.dedent(f"""\
            Generate professional multi-hop QA from:

            Context:
            {text}

            Strict requirements:
            1. Extract exactly 2-3 interrelated facts
            2. Question must demonstrate cross-fact reasoning
            3. Use this exact JSON structure (include all quotes/braces):
            {{
                "question": "...",
                "reasoning_steps": [
                    {{"step": "Must explicitly use Fact 1"}},
                    {{"step": "Must explicitly link Fact 2"}}
                ],
                "answer": "...",
                "supporting_facts": ["Verbatim Fact 1", "Verbatim Fact 2"],
                "type": "..."
            }}
            """)
        else:
            user_prompt = textwrap.dedent(f"""\
                请基于以下上下文生成专业级多跳问答：

                上下文：
                {text}

                严格按照以下要求执行：
                1. 必须从上述上下文中提取2-3个关联事实
                2. 问题需体现跨事实推理的复杂性
                3. 使用此精确JSON结构（包括所有引号和括号）：
                {{
                    "question": "...",
                    "reasoning_steps": [
                        {{"step": "必须明确引用事实1"}},
                        {{"step": "必须明确关联事实2"}}
                    ],
                    "answer": "...",
                    "supporting_facts": ["事实1原文", "事实2原文"],
                    "type": "..."
                }}
            """)
        
        return user_prompt