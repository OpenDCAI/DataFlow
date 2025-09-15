import json

class Doc2QAAutoPromptGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, seed_data: str) -> str:
        prompt = f'''You will be given a piece of seed data, which may consist of a paragraph, dialogue, or any other form of text containing potential question-answer information.
Your task is to analyze this seed data carefully and generate a clear and effective prompt that can be used to instruct a language model to extract a single high-quality question-answer (QA) pair suitable for reinforcement learning (RL) training from this piece of data.

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
Don't output the formatting instructions, just the prompt itself.
Here is the seed data you need to analyze and generate a prompt for:\n{seed_data}'''

        return prompt


class Doc2QAQuestionQualityPrompt:
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

class Doc2QAAnswerAlignmentPrompt:
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

class Doc2QAAnswerVerifiabilityPrompt:
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

class Doc2QADownstreamValuePrompt:
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