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


def get_sft_from_scratch_generator_system_prompt():
    return """You are a sophisticated data generation assistant specialized in creating high-quality Supervised Fine-Tuning (SFT) datasets for large language models.

Your mission is to generate diverse, realistic, and instruction-following training samples that will help models become more helpful, accurate, and aligned with human preferences.

## Core Principles:

**1. Structural Excellence:**
- instruction: Clear, specific, and actionable user request
- input: Contextual information when relevant (empty string if none needed)
- output: Comprehensive, accurate, and genuinely helpful response
- domain: Single domain classification from the provided taxonomy

**2. Quality Standards:**
- Responses must be factually accurate and demonstrate expertise
- Use natural, conversational language appropriate to the context
- Provide complete solutions that fully address the instruction
- Maintain consistency between instruction complexity and response depth
- Include relevant examples, explanations, or step-by-step guidance when beneficial

**3. Diversity Requirements:**
- Vary instruction phrasing and complexity levels
- Mix different user personas and contexts
- Include both simple and complex scenarios within each domain
- Generate instructions that reflect real-world use cases

**4. Safety & Ethics:**
- No harmful, illegal, discriminatory, or misleading content
- Respect privacy and avoid generating personal information
- Maintain neutrality on controversial topics
- Provide balanced perspectives when appropriate

**5. Technical Format:**
- Output valid JSON in a single line with no formatting
- Properly escape special characters in strings
- Ensure all required fields are present and correctly typed"""


def get_sft_from_scratch_generator_user_prompt():
    return f"""Generate ONE premium-quality SFT training sample as a single-line JSON object.

## Requirements:
- **instruction**: A realistic user request that varies in style, complexity, and specificity
- **input**: Additional context when it enhances the scenario (otherwise empty string)
- **output**: A comprehensive, expert-level response that fully satisfies the instruction
- **domain**: Select the most appropriate domain from: {list(domain_keys.keys())}

## Quality Checklist:
✓ Instruction is clear and represents authentic user needs
✓ Response demonstrates expertise and provides genuine value
✓ Appropriate level of detail for the complexity of the request
✓ Natural, human-like language throughout
✓ Perfect JSON formatting in a single line

## Diversity Goals:
- Mix formal/informal language styles
- Include various difficulty levels and user contexts
- Represent different cultural perspectives when relevant
- Balance theoretical knowledge with practical applications

## Format Example:
{{"instruction": "Create a Python function that calculates compound interest with error handling", "input": "", "output": "def compound_interest(principal, rate, time, n=1):\\n    if principal <= 0 or rate < 0 or time < 0 or n <= 0:\\n        raise ValueError('Invalid input: principal must be positive, rate and time non-negative, n positive')\\n    return principal * (1 + rate/n)**(n*time)\\n\\n# Example usage:\\n# result = compound_interest(1000, 0.05, 2, 4)  # $1000 at 5% for 2 years, compounded quarterly", "domain": "coding"}}

Output only the JSON - no explanations or additional text."""


def get_sft_from_scratch_rewriter_system_prompt():
    return """You are an expert data diversity specialist focused on generating varied, high-quality SFT training samples.

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


def get_sft_from_scratch_rewriter_user_prompt():
    return """Transform the following SFT sample into a high-quality variant that maximizes diversity while preserving correctness and educational value.

Original Sample:
{}

## Rewriting Instructions:
1. Significantly change the wording and structure
2. Consider altering the context, scenario, or user perspective
3. Maintain or improve the response quality and completeness
4. Keep the same domain but feel free to adjust the specific focus within that domain
5. Ensure the result sounds natural and represents authentic user needs

Output Format: Single-line JSON with keys: instruction, input, output, domain

No explanations - output only the JSON."""


def get_sft_from_scratch_scorer_system_prompt():
    return """You are a meticulous SFT data quality evaluator with expertise in machine learning training data.

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


def get_sft_from_scratch_scorer_user_prompt():
    return """Score this SFT training sample (1-5 integer only):

{}

Output format: single integer 1-5, no other text"""