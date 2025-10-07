import json

class RAREDoc2QueryGenertorPrompt:
    '''
    The prompt for the Doc2Query Generator.
    '''

    def __init__(self):
        pass

    def build_prompt(self, document) -> str:
        prompt = '''# Context
You are tasked with generating reasoning-intensive questions with scenarios based on a given document. These questions must be standalone (meaningful without the document) while being answerable using information from the document as supporting evidence. The questions should specifically engage with core concepts and principles from the document's domain.

# Question Requirements
1. Each question MUST:
- Present a complete scenario or context within itself
- Be answerable through logical reasoning and critical thinking
- Remain valid and meaningful even if the source document didn't exist
- Target higher-order thinking skills (analysis, evaluation, synthesis)
- Be domain-relevant but not document-specific
- Incorporate key concepts, terminology, and principles from the document's field
- Challenge understanding of domain-specific problem-solving approaches

2. Each question MUST NOT:
- Directly reference the document or its contents
- Be answerable through simple fact recall
- Require specific knowledge only found in the document
- Be a reading comprehension question
- Stray from the core subject matter of the document's domain

# Domain Alignment Guidelines
Before generating questions:
1. Identify the primary domain (e.g., programming, medicine, economics)
2. Extract key concepts and principles from the document
3. List common problem-solving patterns in this domain

When crafting questions:
1. Frame scenarios using domain-specific contexts
2. Incorporate relevant technical terminology naturally
3. Focus on problem-solving approaches typical to the field
4. Connect theoretical concepts to practical applications within the domain

After generating questions step by step, reformat questions including corresponding scenarios in JSON with key "hard_query":
```json
{{
    "hard_query": { "question": <str>, "scenario": <str>}
}}
```
Now, ** the number of hard_queries to generate is exactly 1 **.
        # Document
        '''
        prompt_document = document
        return prompt + prompt_document


class RAREReasonDistillGenertorPrompt:
    '''
    The prompt for the ReasonDistill Generator.
    '''

    def __init__(self):
        pass

    def build_prompt(self, scenario,question,documents_str) -> str:
        prompt = f'''# Scenario
{scenario}

# Question
{question}

# Retrieved Documents
{documents_str}

# Instructions:
1. Identify the essential problem.
2. Identify the helpful information to address the questions. Not all retrieved documents are relevant.
3. Think step by step to reason and draft an answer with as many thoughts as you have.
        '''
        return prompt
