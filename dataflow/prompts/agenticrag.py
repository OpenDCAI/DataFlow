import json
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
"""
A collection of prompts for the AgenticRAG pipelines operator
"""

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorGetIdentifierPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get identifier.
    '''
    def __init__(self):
        pass
    
    def build_system_prompt(self) -> str:
        system_prompt = '''
        You need to extract the content_identifier from question. Here's how:
  1. For each question, identify the main subject/noun phrase that the question is about
  2. This should typically be:
    - Proper nouns (names, titles)
    - Specific technical terms
    - Unique identifiers in the question

  Examples:
  {
      "question": "What is the third movie in the Avatar series?",
      "content_identifier": "Avatar series"
  },
  {
      "question": "龙美术馆2025年展览展览时间范围是什么",
      "content_identifier": "龙美术馆"
  }

  Return JSON format with key "content_identifier"        
'''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
        Now process this question:{input}
        '''
        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorGetConlcusionPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get initial conclusion.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
  # Conclusion Extraction and Relationship Generation Specifications

  ## I. Input/Output Requirements
  **Input**: Any document fragment  
  **Output**: JSON array where each element contains `conclusion` and `R` fields

  ## II. Conclusion Extraction Rules
  1. **Atomicity**  
      - Each conclusion must be an indivisible basic fact  
      - ✖ Prohibited combined conclusions: "A increased by 5% and B decreased by 2%" → Should be split into two conclusions

  2. **Verifiability**  
      - Must contain at least one definite identifier:  
        ✓ Numeric value (59.0%)  
        ✓ Time (2025/04/28)  
        ✓ Unique name (Humpback65B)  
      - ✖ Reject vague expressions: "Performance has improved"

  3. **Timeliness Handling**  
      - Explicitly mark time ranges when containing time-sensitive information  
      - Examples:  
        ✓ "Global GDP grew by 3.0% in 2023"  
        ✖ "Recent GDP growth of 3.0%"

  4. **Citation Integrity**  
      - If a conclusion cites other content (e.g., "as stated in (2)"), the complete content of (2) must be embedded in the conclusion

  ## III. Relationship (R) Generation Standards
  ### Attribute Requirements
  - **Structured**: Use semicolons to separate multi-metrics (Example 3)  
  - **Operational**: Directly usable for database queries or calculations  
    ✓ "City with the highest temperature"  
    ✖ "Conclusions about temperature"

  ### Generation Templates
  | Conclusion Type         | R Template                            | Example                         |
  |-------------------------|---------------------------------------|---------------------------------|
  | Single Numeric Result   | "[Indicator Name]"                    | A: "59.0%" → R: "Accuracy"      |
  | Comparative Conclusion  | "[Indicator] compared to [baseline] in [change dimension]" | A: "4.2% higher than baseline" → R: "Improvement in accuracy compared to baseline" |
  | Multi-dimensional Result| "[Primary Indicator] and its [sub-dimension] distribution" | A: "Average 59% (Humanities 65.6%)" → R: "Average accuracy and subject distribution" |

  ## IV. Output Specifications and Examples
  [
    {
      "conclusion": "Humpback65B achieved a zero-shot accuracy of 59.0% in the MMLU evaluation",
      "R": "Humpback65B's zero-shot accuracy"
    },
    {
      "conclusion": "On 2025/04/28, the closing price of XL Er Nantes-U was $11.34 (up 14.0%)",
      "R": "Closing price and percentage increase of XL Er Nantes-U on 2025/04/28"
    },
    {
        "conclusion": "90% of 27 million metric tons",
        "R": "Proportion of new global LNG supply from North America in 2025"
    },
    {
        "conclusion": "Abstract",
        "R": "Indexed part of Springer articles in databases"
    },
    {
        "conclusion": "2024-03-06",
        "R": "Publication date of Psychology Top 100 of 2023"
    },
    {
        "conclusion": "2018-01",
        "R": "Collection date of 'The Importance of Referencing - PMC'"
    },
    {
        "conclusion": "30-40%",
        "R": "Percentage of science report dedicated to results section"
    },
    {
        "conclusion": "$500 billion",
        "R": "Projected economic contribution of hybrid work models by 2030"
    },
    {
        "conclusion": "650,000+",
        "R": "Number of youth insights in India Skills Report 2025"
    },
    {
        "conclusion": "July 2024 issue",
        "R": "Consumer Reports publication in July 2024"
    },
    {
        "conclusion": "16th annual, 2024-12",
        "R": "Edition and publication date of Deloitte's Tech Trends 2025"
    },
    {
        "conclusion": "January 2024 issue",
        "R": "Consumer Reports publication in January 2024"
    },
    {
        "conclusion": "December 2021 issue",
        "R": "Consumer Reports publication in December 2021"
    },
    {
        "conclusion": "November 2021 issue",
        "R": "Consumer Reports publication in November 2021"
    },
    {
        "conclusion": "14",
        "R": "Death count in listeria outbreak linked to frozen shakes"
    },
    {
        "conclusion": "$122 million",
        "R": "Mega Millions jackpot amount for May 16"
    },
    {
        "conclusion": "32",
        "R": "Number of consecutive years United Way met its goal"
    },
    {
        "conclusion": "62%",
        "R": "Percentage increase in Chemical Sciences article submissions (2014-2021)"
    },
    {
        "conclusion": "11 pounds of fish",
        "R": "Fish trade for Europa League semifinal ticket"
    },
    {
        "conclusion": "2-1",
        "R": "PSG vs. Arsenal match result (Champions League)"
    }
  ]
        '''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
    The document content to be processed is as follows: {input}
    '''
        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorQuestionPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get initial question.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''Your task is to generate a corresponding question (Q) based on the given task identifier (ID), relationship (R), and answer (A).

  Input/Output Specifications:
  Input:
  - ID: Data source or query scope
  - R: Logical relationship for extracting the answer from the data
  - A: Known correct answer

  Output:
  - Must be in strict JSON format: {"Q": "generated question"}
  - No explanations or extra fields allowed

  Q must satisfy:
  1. Be a complete natural language question
  2. Allow deriving answer A by applying R after accessing context via ID

  Question Generation Principles:
  1. Exact correspondence - Each question must fully base on the original conclusion, with the answer being its core content.
  2. Derivability - The original conclusion must be directly derivable from the question and be the only correct answer.
  3. Self-containment - Questions must be complete and independent, not relying on external references or unspecified context.
  4. Information hiding - Do not reveal specific sources or data paths, but can include search hints.
  5. Specificity and clarity - Questions should include details like specific times to ensure unique answers.
  6. Single question - Generate only one question per conclusion.
  7. If the conclusion can only be obtained from input content, include hints via data source identifiers in the question.
  8. Language consistency - The language of each question must be the same as the conclusion's language.

  Examples:
  Input:
  ID: Global daily maximum temperatures
  R: City with the highest temperature
  A: xx City
  Output: {"Q": "What is the city with the highest temperature in global daily maximum temperatures?"}

  Only output JSON without additional content.
  '''
        return system_prompt
    
    def build_prompt(self, identifier, conclusion, relation) -> str:
        prompt = f'''
        Data to be Processed:
        ID: {identifier}
        R: {relation}
        A: {conclusion}
        '''

        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorCleanQAPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to clean QA.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''Processing Rules:
  1. Extract ONLY the exact information requested in the question
  2. Preserve the original index numbering
  3. Never omit essential information
  4. Standardize all numerical formats:
      - Percentages: 8% (not "8percent" or "eight percent")
      - Numbers: Use commas for thousands (3,045)
      - Currency: $1,000 (not "1000 dollars")
      - Dates: YYYY-MM-DD format
      - Units: include (5kg, 10cm, etc.)

  Example:
  {
      "question": "How many travel trends for 2022 does '2025 Annual Travel Trends Report' present?",
      "original_answer": "The Neo4j graph database was used to organize 3,045 Raman spectra of exosomes.",
      "refined_answer": "3,045"
  }

  Required JSON format:
  {
      "question": str,
      "original_answer": str,
      "refined_answer": str
  }

  Key requirements:
  - Be extremely concise in refined_answer
  - Never add information not present in original_answer
  - Preserve all numerical values exactly
  - If question asks for specific data, extract only that data
  '''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
            The data need to be processed is as follows: {input}
        '''

        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorAnswerPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get LLM's answer.
    '''
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        prompt = f'''Please solve the following problem and return as many relevant results as possible that meet the query requirements.\n Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.\n The task is:\n {input}
        '''.strip()
        
        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorRecallScorePrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get recall score.
    '''
    def __init__(self):
        pass

    def  build_system_prompt(self) -> str:
        system_prompt = '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria 
    1) 2 points: the information between the golden answer and the other answer completely consistent, although the expression methods can be different. 
    2) 1 point: the other answer contains all the information of the golden answer but has additional valid information.
    3) 0 point: the other answer lacks the necessary key information of the golden answer, or there are contradictions in both the information.
  
  # Examples:
    1) Examples for 2 points: 
        1.1) two answers are completely consistent:
            - Golden answer: Interest rates should be raised and inflation should be monitored.
            - Other answer: It is necessary to raise interest rates and monitor inflation.
    2) Examples for 1 point: 
        2.1) the other answer contains all the information of the golden answer and adds extra useful information:
        - Golden answer: The interest rates should be raised.
        - Other answer: The interest rates should be raised and inflation should be monitored.
    3) Examples for 0 point: 
        3.1) the other answer lacks the key information of the golden answer:
        - Golden answer: The interest rates should be raised and inflation should be monitored.
        - Other answer: The interest rates should be raised.
        3.2) the other answer has contradictions:
        - Golden answer: Interest rates should be raised by 50 basis points.
        - Other answer: Interest rates should be raised by 25 basis points.
  
  # the output should be in JSON format as required without any irrelevant content
  {
    "answer_analysis":"give out the reason on how to score the llm_answer",
    "answer_score":0/1/2
  }
'''
        return system_prompt
    
    def build_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorOptionalAnswerPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get optional answer.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = """
  You are an expert in **linguistic variation** and **data augmentation**. Your task is to generate a comprehensive list of all plausible and commonly recognized alternative expressions, formats, and aliases for a given input entity or piece of information. The goal is to create high-quality training data that captures diverse ways of referring to the same concept.

  **Key Guidelines:**

  1.  **Equivalence:** Each alternative expression must refer to *exactly the same entity or information* as the original input. Do not include broader categories, narrower sub-types, or related but distinct concepts.
  2.  **Scope of Variation:** Focus on:
      Different **formatting conventions** (e.g., dates, numbers, units).
      Common **abbreviations, acronyms, or initialisms**.
      Well-known **aliases, nicknames, or shorter forms** in common usage.
      Synonyms or rephrasing should *only* be included if they are direct, commonly accepted equivalents.
  3.  **Context-Agnosticism:** Unless the input itself implies a specific context, generate general-purpose variations. Avoid creating variations that are only valid in very niche or obscure contexts.
  4.  **Inclusion of Original:** Always include the original input as the first item in the generated list.
  5.  **Format:** Output the variations as a JSON list of strings.

  **Examples:**

  Input: 1977-01-26
  Output: ["1977-01-26", "1977 01 26", "1977.01.26", "January 26, 1977", "26 Jan 1977", "Jan 26, 1977"]

  Input: United Nations
  Output: ["United Nations", "U.N.", "UN"]

  Input: 3.14159
  Output: ["3.14159", "π", "pi", "PI"]

  Input: Doctor of Philosophy
  Output: ["Doctor of Philosophy", "Ph.D.", "PhD", "Doctorate"]

  Input: New York City
  Output: ["New York City", "NYC", "The Big Apple"]

  Input: kilogram
  Output: ["kilogram", "kg", "kilograms"]
        
        """

        return system_prompt

    def build_prompt(self, answer) -> str:
        prompt = f"""
    The original answer is: {answer}
    Please list all possible textual expressions that have the same meaning or refer to the same entity, especially in different formats (e.g., dates, names, abbreviations).
    Respond with a JSON list of strings. Do not explain.

        """
        return prompt

@PROMPT_REGISTRY.register()
class AtomicTaskGeneratorGoldenDocAnswerPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get LLM's answer with golden doc.
    '''
    def __init__(self):
        pass

    def build_prompt(self, golden_doc, question) -> str:
        prompt = f"""You are given the following document that contains relevant information to help answer a question.
Document:
\"\"\"
{golden_doc}
\"\"\"
Question:
{question}
Please answer the question using ONLY the information in the provided document. Return the final answer directly, with no explanation.
        """
        return prompt
    
@PROMPT_REGISTRY.register()
class DepthQAGeneratorGetIdentifierPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get identifier.
    '''
    def __init__(self):
        pass
    
    def build_system_prompt(self) -> str:
        system_prompt = '''
        You need to extract the content_identifier from question. Here's how:
  1. For each question, identify the main subject/noun phrase that the question is about
  2. This should typically be:
    - Proper nouns (names, titles)
    - Specific technical terms
    - Unique identifiers in the question

  Examples:
  {
      "question": "What is the third movie in the Avatar series?",
      "content_identifier": "Avatar series"
  },
  {
      "question": "龙美术馆2025年展览展览时间范围是什么",
      "content_identifier": "龙美术馆"
  }

  Return JSON format with key "content_identifier"        
'''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
        Now process this question:{input}
        '''
        return prompt

@PROMPT_REGISTRY.register()
class DepthQAGeneratorBackwardTaskPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get backward task.
    '''
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        prompt = f'''
        Conduct divergent searches based on the input element to find an appropriate superset related to its attributes, and elaborate on the relationship between the superset and the element (mine for special and uniquely pointing relationships to ensure that the superset + relationship does not mislead to other subsets). Example supersets include:
  1. The superset of a paragraph or sentence can be the text content it belongs to.
  2. The superset of a specific term can be its corresponding discipline or category.
  3. The superset of a specific date can be any date range containing it, such as the week or month it belongs to.
  4. The superset of a short event can be the complete specific event it belongs to.
  5. The superset of a page can be other pages referencing it or its parent page.
  6. Only generate one relationship, and the content of the relationship should preferably not include strongly specific proper nouns.
  
  Optional expressions for relationships:
  1. Clearly express hierarchical or ownership relationships. If the input is a sub-item of a series of works, the relation should indicate its position; if the input is a part of a superset, the relation should clarify its ownership.
  2. Provide the specific positioning of the input content, such as time range, field of paper publication, or specific role in the superset.
  3. Wording should conform to the research field or industry standards of the input content.
  4. Only provide necessary association information to avoid irrelevant content. Good example: "This study is part of the IRAM NOEMA Large Program research collection". Bad example: "This study is a very important research conducted by many scientists and has produced very meaningful results" (verbose and containing subjective evaluations).
  
  Note:
  1. Please return the identifier of the superset content, such as attribute name, web page title, paper title, etc., which uniquely locates the superset content.
  2. The content of the superset needs to be obtained through tool invocation, which can be specific web content, PDF text, or image understanding content.
  3. Please clearly describe the relationship between the superset content and the input element, that is, list the qualification conditions from the superset content to ensure that the conditions uniquely point to the input element, and the description of the conditions should be concise.
  4. Use a maximum of 3 search keywords per search; if more than 3 keywords are needed, perform multiple searches separately.
  5. The obtained identifier should preferably be derived from search results and not include the input content.
  6. If the input is a PDF document, give priority to invoking tools to read the document content.
  
  Return format requirements: Please return the result in JSON format with keys 'identifier': str (identifier) and 'relation': str (relationship).
  
  Here are some reference input-output examples: 
  Example1:
  Input: Avatar 3: Fire and Ash
  identifier: Avatar film series
  relation: The third film
  
  Example2:
  Input: The 15 social media trends that will shape your 2025 strategy
  identifier: Hootsuite blog end of 2024
  relation: The authoritative trends report published by Hootsuite to guide social media strategy development

  Example3:
  Input: SOLIS (Seeds of Life In Space) project
  identifier: NOEMA Large Program
  relation: A sub-project within NOEMA's specific large observation program related to research on the existence of life in the universe.
  
  Example4:
  Input: SOLIS. XIX. The chemically rich SVS13-B protostellar jet
  identifier: IRAM NOEMA Large Program research collection
  relation: One of the imaged enriched molecular jet samples in the IRAM NOEMA Large Program research collection, specifically imaged and analyzed for molecular distribution and composition within the collection, uniquely locatable via observation data on SVS13-B in the collection.
  
  Example5:
  Input: AdCare -VLM: Leveraging Large Vision Language Model (LVLM) to Monitor Long-Term Medication Adherence and Care
  identifier: A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges
  relation: A paper that introduces advancements in large vision language models in A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges, covering models including the LVLM described in the input paper.
  
  Example6:
  Input: Immigration is a higher priority for Americans in 2025: AP-NORC poll | AP News
  identifier: 2025 policy priorities report for AAPI communities
  relation: The poll results about shifting immigration priorities featured in AP News and referenced in AAPI policy reports

  Example7:
  Input: X-ray Absorption Spectroscopy (XAS) database for iron-containing proteins (arXiv:2504.18554)
  identifier: iron-binding proteins database
  relation: The specialized database that collects XAS data specifically for proteins containing iron

  Example8: 
  Input: live-action 'Snow White' movie controversy
  identifier: Disney animated film adaptation
  relation: The controversial live-action movie adapted from a Disney animated film featuring the main character Snow White
  
  Example9:
  Input: Evaluating the evidence: a systematic review of reviews of the effectiveness and safety of digital interventions for ADHD | BMC Psychiatry | Full Text
  identifier: BMC Psychiatry journal 2025 publications
  relation: The full-text systematic review about digital ADHD interventions published in BMC Psychiatry
  
  Example10:
  Input: Enron Corporation
  identifier: 2001 Fortune Global 500 energy industry rankings
  relation: The company that ranked first in revenue in the energy sector according to the 2001 Fortune Global 500 rankings
  
  Current input: 
  {input}
        '''
        return prompt

@PROMPT_REGISTRY.register()
class DepthQAGeneratorSupersetCheckPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to check superset.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
**Task**: Validate if a given "superset" can uniquely identify a "subset" based on the provided "relationship".  
  
  **Rules**:  
  1. **Superset-Subset Relationship**:  
    - The "superset" must be a true generalization of the "subset" (e.g., "Animal" is a valid superset of "Dog").  
    - The "superset" CANNOT be a synonym of the "subset" (e.g., "Car" and "Automobile" are invalid).  
  
  2. **Relationship Validity**:  
    - The relationship must **explicitly and uniquely** link the superset to the subset.    
    - It CANNOT be a **many-to-one mapping**.  
  
  **Output Format**:  
  Return a JSON with the key `new_query`. The value should be:  
  - `"valid"` if the superset and relationship can uniquely locate the subset.  
  - `"invalid"` otherwise.  
  
  **Example Valid Output**:  
  {"new_query": "valid"}
'''
        return system_prompt
    
    def build_prompt(self, new_id, relation, identifier) -> str:
        prompt = f'''
Given superset: {new_id}\n
Given relationship: {relation}\n
Given subset: {identifier}\n
'''
        return prompt

@PROMPT_REGISTRY.register()
class DepthQAGeneratorQuestionPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get question.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
  Please generate a question based on the content of the input identifier, a certain answer, and a certain relationship (this relationship is the relationship between the content of the file corresponding to the identifier and the given answer), such that
  The answer to this question is the input answer.
  The content of this question is determined by the content of the identifier and the content of the given relationship.
  The generated question should not involve the content of the input answer.
  Please return it in JSON format, with the key of the JSON being new_query.
'''
        return system_prompt
    
    def build_prompt(self, new_id, relation, identifier) -> str:
        prompt = f'''
                Certain answer: {identifier}\n
                Identifier: {new_id}\n
                Relationship: {relation}\n
'''
        return prompt

@PROMPT_REGISTRY.register()
class DepthQAGeneratorAnswerPrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get LLM's answer.
    '''
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        prompt = f'''
Please solve the following problem and return as many relevant results as possible that "
"meet the query requirements. Ensure responses are as concise as possible, focusing only "
"on key information while omitting redundant details."
"Please return the result in JSON format with keys 'answer_list': List[str] the list of answers."
"\n\n"
"The task is: \n
{input}
        '''.strip()
        
        return prompt

@PROMPT_REGISTRY.register()
class DepthQAGeneratorRecallScorePrompt(PromptABC):
    '''
    The prompt for the AtomicTaskGenerator to get recall score.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria 
    1) 2 points: the information between the golden answer and the other answer completely consistent, although the expression methods can be different. 
    2) 1 point: the other answer contains all the information of the golden answer but has additional valid information.
    3) 0 point: the other answer lacks the necessary key information of the golden answer, or there are contradictions in both the information.
  
  # Examples:
    1) Examples for 2 points: 
        1.1) two answers are completely consistent:
            - Golden answer: Interest rates should be raised and inflation should be monitored.
            - Other answer: It is necessary to raise interest rates and monitor inflation.
    2) Examples for 1 point: 
        2.1) the other answer contains all the information of the golden answer and adds extra useful information:
        - Golden answer: The interest rates should be raised.
        - Other answer: The interest rates should be raised and inflation should be monitored.
    3) Examples for 0 point: 
        3.1) the other answer lacks the key information of the golden answer:
        - Golden answer: The interest rates should be raised and inflation should be monitored.
        - Other answer: The interest rates should be raised.
        3.2) the other answer has contradictions:
        - Golden answer: Interest rates should be raised by 50 basis points.
        - Other answer: Interest rates should be raised by 25 basis points.
  
  # the output should be in JSON format as required without any irrelevant content
  {
    "answer_analysis":"give out the reason on how to score the llm_answer",
    "answer_score":0/1/2
  }
'''
        return system_prompt
    
    def build_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt
    
@PROMPT_REGISTRY.register()
class WidthQAGeneratorMergePrompt(PromptABC):
    '''
    The prompt for the WidthQAGenerator to merge prompt.
    '''
    def __init__(self):
        pass
    
    def build_system_prompt(self) -> str:
        system_prompt = '''
        # Comprehensive Task Guide for Research Questions

  ## Core Objective:
  Intelligently merge 2-3 related research questions into high-quality comprehensive questions while maintaining the integrity and accuracy of the original content.

  ## Input Requirements:
  - Each question includes: index (unique ID), question (question text), golden_answer (standard answer), content_identifier (content identifier)

  ## Grouping Specifications:

  ### Grouping Strategies:
  1. **Content Matching Principle**:
     - Priority: Merge questions with similar themes

  2. **Quantity Control**:
     - Each group must contain 2-3 original questions
     - Ensure all original questions are grouped (no omissions)

  ### Standards for Question Synthesis:
  1. **Content Integrity**:
     - Retain all elements of the original questions
     - Do not add new facts or assumptions
     - Completely preserve time-related elements in their original form

  2. **Question Quality**:
     - Clear and unambiguous expression
     - Logically coherent merged questions
     - Do not imply any solution methods

  3. **Structural Requirements**:
     - Form complete interrogative sentences (not simply connected with "and")
     - Correct grammatical structure
     - Preserve professional terminology in its original form

  ## Output Specifications:
  [
    {
      "question": "Text of the synthesized question",
      "index": [1,2,3], // Original indices
      "content_identifier": "Original content identifier"
    }
  ]
        '''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
        Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), question (original question), and content_identifier (identifier).
'''
        return prompt

@PROMPT_REGISTRY.register()
class WidthQAGeneratorOriginCheckPrompt(PromptABC):
    '''
    The prompt for the WidthQAGenerator to check origin.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
    Task Instructions:
  Verify if complex questions can be properly decomposed into their original questions.
  Return state=1 if all conditions are met, state=0 otherwise:

  Conditions for state=1:
  1. The complex question clearly contains all elements from original questions
  2. No information distortion or ambiguity introduced
  3. Logical relationships between original questions are properly maintained


  For example:  
  "index": 1  
  "Complex Question": "In the Academia Insider article 'The best AI tools for research papers and academic research (Literature review, grants, PDFs and more)', how does Semantic Scholar enhance literature review efficiency? Who are the two contributors—one with a Master’s and Ph.D. in Chemistry from the UK and Australia, and the other a Ph.D. student at Simon Fraser University (SFU)—credited with contributing academic insights and initiating the list of AI research tools, respectively?"  
  "Original Questions": [  
      "According to 'The best AI tools for research papers and academic research (Literature review, grants, PDFs and more) - Academia Insider', how does Semantic Scholar enhance literature review efficiency?",  
      "In the Academia Insider article 'The best AI tools for research papers and academic research (Literature review, grants, PDFs and more)', who is the contributor with a Master’s and Ph.D. in Chemistry from the UK and Australia and extensive research experience?",  
      "In the Academia Insider article 'The best AI tools for research papers and academic research (Literature review, grants, PDFs and more)', who is the contributor credited with helping to start the list of AI research tools?"  
  ]  
  The above complex question can be decomposed into these original questions without deviation in content, and the status is returned as 1.  

  "index": 2  
  "Complex Question": "Based on the trends reported in the 2025 scientific publications of the Academy of Articles and the information on open and free content from the JSTOR and Artstor 'About JSTOR' page, when does research on protecting cultural and linguistic diversity through AI reach its peak? What is the total number of research reports available, and how many policy institutes are represented in the collection?"  
  "Original Questions": [  
      "According to the 2025 scientific publication trends of the Academy of Articles, when does research on protecting cultural and linguistic diversity through AI reach its peak?",  
      "According to the information on open and free content from the JSTOR and Artstor 'About JSTOR' page, what is the total number of research reports in the collection? How many policy institutes are covered?"  
  ]  
  The above complex question cannot be decomposed into original questions because the direction of the questions in the complex question is confusing and ambiguous, and the status is returned as 0.

  Example Output:
  [{
      "index": 1,
      "complex_question": "original complex question",
      "state": 1
  }]
'''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
    Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), complex_question (original complex question), 
    and original_questions (list of original questions).
'''
        return prompt

@PROMPT_REGISTRY.register()
class WidthQAGeneratorQuestionVerifyPrompt(PromptABC):
    '''
    The prompt for the WidthQAGenerator to verify question.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
  Answer the provided complex research questions based on your knowledge.
  For each question, provide your answer.

  Output JSON format:
  [{
  "index": 1 // original question indices
  "complex_question": original complex question,
  "llm_answer"://your answer

  },
  {
  "index": 2 // original question indices
  "complex_question": original complex question,
  "llm_answer"://your answer
  }]
'''
        return system_prompt
    
    def build_prompt(self, input) -> str:
        prompt = f'''
    Please answer these research questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''
        return prompt

@PROMPT_REGISTRY.register()
class WidthQAGeneratorAnswerPrompt(PromptABC):
    '''
    The prompt for the WidthQAGenerator to get LLM's answer.
    '''
    def __init__(self):
        pass

    def build_prompt(self, input) -> str:
        prompt = f'''
Please solve the following problem and return as many relevant results as possible that "
"meet the query requirements. Ensure responses are as concise as possible, focusing only "
"on key information while omitting redundant details."
"Please return the result in JSON format with keys 'answer_list': List[str] the list of answers."
"\n\n"
"The task is: \n
{input}
        '''.strip()
        
        return prompt

@PROMPT_REGISTRY.register()
class WidthQAGeneratorRecallScorePrompt(PromptABC):
    '''
    The prompt for the WidthQAGenerator to get recall score.
    '''
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = '''
Evaluate the consistency of the core content of the golden answer and the other answer
  # Scoring Criteria 
    1) 2 points: the information between the golden answer and the other answer completely consistent, although the expression methods can be different. 
    2) 1 point: the other answer contains all the information of the golden answer but has additional valid information.
    3) 0 point: the other answer lacks the necessary key information of the golden answer, or there are contradictions in both the information.
  
  # Examples:
    1) Examples for 2 points: 
        1.1) two answers are completely consistent:
            - Golden answer: Interest rates should be raised and inflation should be monitored.
            - Other answer: It is necessary to raise interest rates and monitor inflation.
    2) Examples for 1 point: 
        2.1) the other answer contains all the information of the golden answer and adds extra useful information:
        - Golden answer: The interest rates should be raised.
        - Other answer: The interest rates should be raised and inflation should be monitored.
    3) Examples for 0 point: 
        3.1) the other answer lacks the key information of the golden answer:
        - Golden answer: The interest rates should be raised and inflation should be monitored.
        - Other answer: The interest rates should be raised.
        3.2) the other answer has contradictions:
        - Golden answer: Interest rates should be raised by 50 basis points.
        - Other answer: Interest rates should be raised by 25 basis points.
  
  # the output should be in JSON format as required without any irrelevant content
  {
    "answer_analysis":"give out the reason on how to score the llm_answer",
    "answer_score":0/1/2
  }
'''
        return system_prompt
    
    def build_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt
    
@PROMPT_REGISTRY.register()
class AtomicQAGeneratorPrompt(PromptABC):
    '''
    The prompt for the AtomicQAGenerator.
    '''
    def __init__(self):
        pass

    def build_prompt(self, gen_qa_num: str, input_doc: str) -> str:
        prompt = f'''You are an information extraction and question generation system.
# Task:
Given a document, extract a set of **atomic, verifiable facts** and convert each into a **QA pair**, where:
- The **question** focuses on a specific, retrievable detail from the document.
- The **answer** is concise, factual, and directly grounded in the document.
- For each document, generate **at most {gen_qa_num} QA pairs**. Prioritize the most concrete, unique, and verifiable facts.
- Only generate questions that require consulting the document to answer — avoid trivial facts or common-sense knowledge.

# Rules for QA Generation

1. Atomicity
- Each QA must be based on a single indivisible fact (no conjunctions).
    ✖ "A increased and B decreased" → must split into two questions.

2. Verifiability
- The answer must include at least one of:
    - ✓ Numeric value (e.g., 59.0%)
    - ✓ Time or date (e.g., 2025/04/28)
    - ✓ Unique name/entity (e.g., Humpback65B)
- ✖ Reject vague expressions: "Performance has improved"

3. Time specificity
- Explicitly mark time ranges when containing time-sensitive information  
- Examples:  
    ✓ "Global GDP grew by 3.0% in 2023"  
    ✖ "Recent GDP growth of 3.0%"

4. Relevance and Precision
- Avoid abstract questions. Focus on measurable and database-friendly details.

5. Answer Uniqueness
- The question must be **specific enough** to yield a **unique answer** from the document.
- ✖ Avoid under-specified questions that allow **multiple correct answers**.
    - ✖ "What awards did Author X receive?" (if multiple awards are listed in the document)
    - ✓ "What award did Author X receive in 2022?" (if only one is given for that year)
- ✖ Avoid vague superlatives like "notable," "important," "significant" without clear criteria.

# Output Format:
A JSON array of QA pairs. Each item contains:
- `question`: A specific, answerable question.
- `answer`: The factual value from the document.

# Examples

```json
[
    {{
    "question": "What is the number 1 sport in the usa?",
    "answer": "American football"
    }},
    {{
    "question": "Where did the Ottoman slave trade flourish?",
    "answer": "In the Balkans"
    }},
    {{
    "question": "Who was president when the white house was built?",
    "answer": "John Adams"
    }}
]
```

The document content to be processed is as follows: 
{input_doc}
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class MergeAtomicQAPrompt(PromptABC):
    '''
    The prompt for the MergeAtomicQAPrompt.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Data: str, New_question: str, New_answer: str, New_document: str) -> str:
        prompt = f'''You are an expert in constructing multi-hop questions grounded in document-based facts.

  ## Task
  You are given multiple question-answer-document triples to generate multi-hop question that require reasoning over the **latest previous hop** (i.e., the final element in Previous_Hops) together with New_pair. Use Previous_Hops strictly as supporting context: they may be consulted to verify entities/constraints and must be preserved (not removed, weakened, or contradicted) by the final question.
  Only produce a multi-hop question if it is logically valid, unambiguous, and well-supported by both documents. If there is any uncertainty or weak connection, return empty JSON list instead of forcing a question.

  ## Input Schema
  Previous_Hops: 
    type: "list of objects (ordered oldest → latest)"
    each_item: 
      - Hop_number
      - Question: string
      - Answer: string
      - Doc: string
  New_pair:
    type: object (This is the candidate QA+Doc to attach to the chain.)
    fields:
      - Question: string
      - Answer: string
      - Doc: string
 
  ## Output Format (ONLY JSON)
  Valid multi-hop case:
  {{
    "type": "inference" | "comparison",
    "final_question": "...",
    "final_answer": "..."
  }}

  If no high quality multi-hop question can be created:
  Return an empty JSON object {{}}

  ## Types
  - inference:
    The final question chains information so that answering it requires (1) the last previous hop's facts and (2) the New_pair's facts. For inference cases, final_answer MUST equal New_pair Answer.
  - comparison:
    The final question compares a shared measurable dimension (e.g., date, numeric quantity, size). Both compared values must be supported explicitly in the documents. The answer should be one of the compared entities.

  ## Rules
  - 1. Use Previous_Hops as supporting context. Do NOT remove, weaken, or contradict any fact, constraint, or entity presented in any Previous_Hops's Question/Answer/Doc.
  - 2. The final question MUST be generated from (latest Previous_Hop) + (New_pair). DO NOT remove or weaken any Previous_Hops's Question important information in final_question
  - 3. DO NOT leak intermediate answers (no explicit exposure of any Previous_Hop answer in the final_question).
  - 4. Do not simply restate the New_pair question, and do not return an unexpanded Hop question that is missing or only partially uses the information from the Previous Hop. The final question MUST depend on all QA-doc pairs to be answerable.
  - 5. Ensure that there is **sufficient and accurate** evidence to get the answer.
  - 6. When linking entities across docs, explicitly verify from the document contents (not just question text) that they refer to the exactly same real-world entity or fact
  - 7. For comparison, only compare facts on the same axis (e.g., date vs date, size vs size), not unrelated attributes.
  - 8. Do not create final questions that just state multiple facts independently, without a reasoning link between them.
  - 9. Favor precision and factual grounding over producing more questions. If the logical connection is weak, ambiguous, or speculative, reject (return {{}}).
  - 10. Output only the final JSON object. Do NOT include chain-of-thought, explanations, or any extra text.

  ## Examples
  ### final_question is inference
  Case 1 (Extend 1-hop question to 2-hop):
  Input:
  Hop_1:
  Question: What is the name of the performer of "Qui de nous deux"?
  Answer: Matthieu Chedid
  Doc: "Qui de nous deux" is performed by Matthieu Chedid.

  New_pair:
  Question: Who is the father of Matthieu Chedid?
  Answer: Louis Chedid
  Doc: Matthieu Chedid is the son of Louis Chedid.

  Good Output:
  {{
    "type": "inference",
    "final_question": "Who is the father of the performer of 'Qui de nous deux'?",
    "final_answer": "Louis Chedid"
  }}

  ✖ Error Case (leaks intermediate answer):
  -"final_question": "Who is the father of Matthieu Chedid?"

  Case 2 (Extend 2-hop inference question to 3-hop):
  Input:
  Hop_1:
  Question: Who is the composer of "Al gran sole carico d'amore"?
  Answer: Luigi Nono
  Doc: "Al gran sole carico d'amore" is an opera with music by Luigi Nono.

  Hop_2:
  Question: Where did the composer of "Al gran sole carico d'amore" work?
  Answer: Venice
  Doc: Luigi Nono was active as a painter in Venice.
  type: inference

  New_pair:
  Question: What is the name of the oldest bridge in Venice?
  Answer: Rialto Bridge
  Doc: The Rialto Bridge is the oldest bridge spanning the Grand Canal in Venice.

  Good Output:
  {{
    "type": "inference",
    "final_question": "What is the name of the oldest bridge in the city where the composer of 'Al gran sole carico d'amore' worked?",
    "final_answer": "Rialto Bridge"
  }}

  ✖ Error Case (leaks intermediate answer):
  -"final_question": "What is the name of the oldest bridge in Luigi Nono worked?"

  Case 3 (Extend 3-hop inference question to 4-hop):
  Input:
  Hop_1:
  Question: Where was Francisco Vázquez born?
  Answer: Guadalajara
  Doc: Francisco H. Vázquez (born June 11, 1949 in Guadalajara, Jalisco, Mexico)

  Hop_2:
  Question: On which continent is Guadalajara located?
  Answer: North America
  Doc: Guadalajara is located in North America.
  type: inference

  Hop_3:
  Question: Who was the Italian navigator who sailed for England and explored the east coast of the continent where Francisco Vázquez was born?
  Answer: John Cabot
  Doc: John Cabot (Italian: Giovanni Caboto; c. 1450 -- c. 1500) was a Venetian navigator and explorer whose 1497 discovery of the coast of North America under the commission of Henry VII of England
  type: inference

  New_pair:
  Question: What is the name of the child of John Cabot?
  Answer: Sebastian Cabot
  Doc: Sebastian Cabot was the son of Italian explorer John Cabot (Giovanni Caboto)

  Good Output:
  {{
    "type": "inference",
    "final_question": "What is the name of the child of the Italian navigator who sailed for England and explored the east coast of the continent where Francisco Vázquez was born?",
    "final_answer": "Sebastian Cabot"
  }}

  ✖ Error Case (leaks intermediate answer):
  -"final_question": "What is the name of the child of the Italian navigator who sailed for England and explored the east coast of North America?"

  ✖ Error Case (remove or weaken important information in Previous_Hops):
  -"final_question": "What is the name of the child of the navigator who explored the east coast of Francisco Vázquez was born?"

  ### final_question is comparison
  Input:
  Question: When was John Beach born?
  Answer: January 1, 1812
  Doc: Major John Beach( January 1, 1812 - August 31, 1874) was a United States Army officer during the Black Hawk and American Civil War.

  New_pair:
  Question: When was Seth Gordon Persons born?
  Answer: February 5, 1902
  Doc: Seth Gordon Persons( February 5, 1902 - May 29, 1965) was an American Democratic politician who was the 43rd Governor of Alabama from 1951 to 1955.

  Good Output:
  {{
    "type": "comparison",
    "final_question": "Who was born first, John Beach or Seth Gordon Persons?",
    "final_answer": "John Beach"
  }}

  Another Good Output:
  {{
    "type": "comparison",
    "final_question": "Was John Beach born before Seth Gordon Persons?",
    "final_answer": "Yes"
  }}

  ✖ Error Case (leaks intermediate answer):
  -"final_question": "Who was born first, John Beach (January 1, 1812 - August 31, 1874) or Seth Gordon Persons?"

  ### ✖ Invalid (spurious linkage: the QA-doc pairs contain unrelated facts that are superficially similar but logically disconnected.)
  Input:
  Hop_1:
  Question: How many cardinals entered the papal conclave on March 31?
  Answer: 27
  Doc: Only twenty-seven cardinals entered the conclave on March 31, 1721.
  
  New_pair:
  Question: Which band did 27 open for in the Czech Republic?
  Answer: Robert Plant
  Doc: 27 is a rock band that opened for Robert Plant in Prague.

  Correct Output:
  {{}}
  ✖ Error Output (In Example: Cardinals and the rock band '27' are unrelated entities):
  final_question: Which band did the cardinals who entered the papal conclave on March 31 open for in the Czech Republic?
  final_answer: Robert Plant

  ### ✖ Invalid (spurious linkage: there is no necessary logical connection between them.)
  Input:
  Hop_1:
  Question: What was the deployment order date for the 16th Army to the Ukraine?
  Answer: 25 May 1941
  Doc: The 16th Army was ordered to deploy to Ukraine on 25 May 1941.

  New_pair:
  Question: Which two spheres of influence were involved in the division of Europe in the 1940s?
  Answer: The Western world and the Soviet Union
  Doc: Postwar Europe was divided into the Western and Soviet spheres of influence.

  Correct Output:
  {{}}
  ✖ Error Output (In Example: No causal or thematic link between army deployment date and geopolitical division):
  final_question: What were the two major spheres of influence following the deployment of the 16th Army to the Ukraine in 1941?
  final_answer: The Western world and the Soviet Union

  ### ✖ Invalid: (entity-based false link: a false connection between facts or documents that arises solely because different entities share identical or highly similar names, without any actual semantic or factual relationship.)
  Input:
  Hop_1:
  Question: Who presents the Statewide Drive program at 107.9 ABC Ballarat?  
  Answer: Nicole Chvastek  
  Doc: "107.9 ABC Ballarat" has a total of 16 full time employees. A breakfast program is presented by Steve Martin from 6.15 am to 10.00 am weekdays. A mornings program is presented by Gavin McGrath from 10.00 am to 11.00 am weekdays. The regional "Statewide Drive" program (3.00 pm to 6.00 pm weekdays) is also broadcast from the Ballarat studios. It is presented by Nicole Chvastek and covers Victoria, southern New South Wales and a small part of eastern South Australia. It does not broadcast into the Melbourne metro area. 107.9 ABC Ballarat, callsign 3CRR, is an ABC Local Radio station.

  New_pair:
  Question: What toolkit has Nicole Joseph designed for breast care?  
  Answer: Breast-CareSolutions toolkit  
  Doc: Nicole Joseph introduced the global and multi-lingual breast care awareness campaign "The Gesture That Saves" in San Francisco in 2016 to 100 global peers from 40 countries during the VV100 retreat. She has designed a comprehensive Breast-CareSolutions toolkit and is currently designing a reproductive-health advocacy program. Nicole Joseph-Chin is the Chief Innovator, Founder and CEO of Ms. Brafit Limited.

  Correct Output:
  {{}}
  ✖ Error Output (In Example: Nicole Chvastek and Nicole Joseph are different individuals):
  final_question: What toolkit has the presenter of the Statewide Drive program at 107.9 ABC Ballarat designed for breast care?  
  final_answer: Breast-CareSolutions toolkit

  ### ✖ Invalid: (trivial concatenation: the final question simply combines facts without needing reasoning or integration.)
  Input:
  Hop_1:
  Question: Where was the State Normal School at Cheney located by the end of the term's first week?  
  Answer: Pomeroy building 
  Doc: By the end of the term's first week, the State Normal School at Cheney was located in the Pomeroy building.

  New_pair:
  Question: Who designed the Cheney Building?
  Answer: H. H. Richardson  
  Doc: The Cheney Building was designed by H. H. Richardson.

  Correct Output:
  {{}}
  ✖ Error Output (In Example: Final_question simply concatenates the two questions together using 'and' without needing reasoning.):
  final_question: Which building was used for the State Normal School at Cheney by the end of the term's first week, and who designed the Cheney Building?
  final_answer: Pomeroy building, H. H. Richardson

  ### Invalid (lacking_evidence: one or both compared values are not explicitly supported in the provided documents with sufficient precision to support the asserted comparison.)
  Hop_1:
  Question: What year was the Cambridge Battery completed for the 100-ton gun?
  Answer: 1886
  Doc: Cambridge Battery was ready in 1886.

  New_pair:
  Question: When did the 1886 United Kingdom general election take place?
  Answer: July 1-27, 1886
  Doc: The 1886 United Kingdom general election took place from 1 July to 27 July 1886.

  Correct Output:
  {{}}
  ✖ Error Output (In Example: The Cambridge Battery is only dated to the year (1886), while the election records include exact dates in July. This lack of precise dating in the document makes it impossible to determine which event occurred first.):
  final_question: Which event occurred first, the completion of the Cambridge Battery in 1886 or the United Kingdom general election led by Charles Stewart Parnell in 1886?
  final_answer: The completion of the Cambridge Battery in 1886.

  ## Checklist
  - [ ] Do all docs describe logically connectable facts?
  - [ ] For comparison: are both facts from the same measurable dimension?
  - [ ] For inference: does the reasoning chain correctly lead to New_pair Answer?
  - [ ] Is the final question truly unanswerable without all QA-doc pairs?
  - [ ] Are all intermediate answers hidden?
  - [ ] Are the linked entities explicitly confirmed as the same in both documents?
  - [ ] Is there sufficient and accurate evidence to get the answer?

  The data need to be processed is as follows:
  {Data}
  
  New_pair:
  Question: {New_question}
  Answer: {New_answer}
  Doc: {New_document}
  Only output the final JSON object. Do not explain your reasoning.
  '''
        return prompt

@PROMPT_REGISTRY.register()
class InferenceCheckPrompt():
    '''
    The prompt for checking the inference multihop question.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Question1: str, Answer1: str, Document1: str, Question2: str, Answer2: str, Document2: str, Final_question: str, Final_answer: str, qa_type: str) -> str:
        prompt = f'''You are a multi-hop QA verification system.
  ## Task
  You are given a multi-hop QA construction based on two question-answer-document triples:
  (Question1, Answer1, Doc1) and (Question2, Answer2, Doc2), and a final multi-hop QA:
  - Final_question
  - Final_answer
  - type: "inference"

  Your job is to **verify whether the final QA is logically valid** according to the reasoning paths and documents.

  ## Input Fields
  - Question1, Answer1, Doc1
  - Question2, Answer2, Doc2
  - Final_question, Final_answer
  - type: "inference"

  ## Output Format
  Return a JSON object:
  {{
    "valid": "true" | "false",
    "error_type": "bad_linkage" | "entity_false_link" | "trivial_concatenation" | "other",
    "justification": "Short explanation of the issue"
  }}

  ## Definitions & Rules
  - "inference": Final question requires combining QA1 and QA2 in a reasoning chain. The final_answer must exactly match Answer2. No intermediate answers should appear in final_question.

  - "bad_linkage": If the two QA-doc pairs contain unrelated facts that are superficially similar but logically disconnected.
  - "entity_false_link": a false connection between two facts or documents that arises solely because different entities share identical or highly similar names, without any actual semantic or factual relationship.
  - "trivial_concatenation": If the final question is formed by simply joining two or more independent facts from the given QA-document pairs into a single sentence (often using "and" or similar conjunctions), without any logical reasoning beyond listing the facts.
  - "other": Other errors that you think are not included above.

  ## Examples

  ### ✓ Valid
  Input:
  Question1: What is the name of the performer of 'Qui de nous deux'?
  Answer1: Matthieu Chedid
  Doc1: "Qui de nous deux" is performed by Matthieu Chedid.

  Question2: Who is the father of Matthieu Chedid?
  Answer2: Louis Chedid
  Doc2: Matthieu Chedid is the son of Louis Chedid.

  Final_question: Who is the father of the performer of 'Qui de nous deux'?
  Final_answer: Louis Chedid
  type: "inference"

  Expected Output:
  {{
    "valid": "true",
    "error_type": "No Error",
    "justification": "Reasoning chain is valid."
  }}

  ### ✖ Invalid (bad linkage)
  Input:
  Question1: How many cardinals entered the papal conclave on March 31?
  Answer1: 27
  Doc1: 27 cardinals entered the 1721 Papal Conclave.

  Question2: Which band did 27 open for in the Czech Republic?
  Answer2: Robert Plant
  Doc2: 27 is a rock band that opened for Robert Plant.

  Final_question: Which band did the cardinals who entered the papal conclave on March 31 open for in the Czech Republic?
  Final_answer: Robert Plant
  type: "inference"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "bad_linkage",
    "justification": "Cardinals and the rock band '27' are unrelated entities"
  }}

  ### ✖ Invalid (bad linkage)
  Input:
  Question1: What was the deployment order date for the 16th Army to the Ukraine?
  Answer1: 25 May 1941
  Doc1: The 16th Army was ordered to deploy to Ukraine on 25 May 1941.

  Question2: Which two spheres of influence were involved in the division of Europe in the 1940s?
  Answer2: The Western world and the Soviet Union
  Doc2: Postwar Europe was divided into the Western and Soviet spheres of influence.

  Final_question: What were the two major spheres of influence following the deployment of the 16th Army to the Ukraine in 1941?
  Final_answer: The Western world and the Soviet Union
  type: "inference"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "bad_linkage",
    "justification": "No causal or thematic link between army deployment date and geopolitical division"
  }}

  ### ✖ Invalid (entity_false_link)
  Input:
  Question1: Who presents the Statewide Drive program at 107.9 ABC Ballarat?  
  Answer1: Nicole Chvastek  
  Doc1: "107.9 ABC Ballarat" has a total of 16 full time employees. A breakfast program is presented by Steve Martin from 6.15 am to 10.00 am weekdays. A mornings program is presented by Gavin McGrath from 10.00 am to 11.00 am weekdays. The regional "Statewide Drive" program (3.00 pm to 6.00 pm weekdays) is also broadcast from the Ballarat studios. It is presented by Nicole Chvastek and covers Victoria, southern New South Wales and a small part of eastern South Australia. It does not broadcast into the Melbourne metro area. 107.9 ABC Ballarat, callsign 3CRR, is an ABC Local Radio station.

  Question2: What toolkit has Nicole Joseph designed for breast care?  
  Answer2: Breast-CareSolutions toolkit  
  Doc2: Nicole Joseph introduced the global and multi-lingual breast care awareness campaign "The Gesture That Saves" in San Francisco in 2016 to 100 global peers from 40 countries during the VV100 retreat. She has designed a comprehensive Breast-CareSolutions toolkit and is currently designing a reproductive-health advocacy program. Nicole Joseph-Chin is the Chief Innovator, Founder and CEO of Ms. Brafit Limited.

  Final_question: What toolkit has the presenter of the Statewide Drive program at 107.9 ABC Ballarat designed for breast care?  
  Final_answer: Breast-CareSolutions toolkit
  type: "inference"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "entity_false_link",
    "justification": "Nicole Chvastek and Nicole Joseph are different individuals"
  }}

  ### ✖ Invalid: (trivial_concatenation)
  Input:
  Question1: Where was the State Normal School at Cheney located by the end of the term's first week?  
  Answer1: Pomeroy building 
  Doc1: By the end of the term's first week, the State Normal School at Cheney was located in the Pomeroy building.

  Question2: Who designed the Cheney Building?
  Answer2: H. H. Richardson  
  Doc2: The Cheney Building was designed by H. H. Richardson.

  Final_question: Which building was used for the State Normal School at Cheney by the end of the term's first week, and who designed the Cheney Building?
  Final_answer: Pomeroy building, H. H. Richardson
  type: "inference"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "trivial_concatenation",
    "justification": "Final_question simply concatenates the two questions together using 'and' without needing reasoning."
  }}

  The data need to be processed is as follows: 
  Question1: {Question1}
  Answer1: {Answer1}
  Doc1:{Document1}
  Question2: {Question2}
  Answer2: {Answer2}
  Doc2:{Document2}
  Final_question:{Final_question}
  Final_answer:{Final_answer}
  type:{qa_type}
  Only return the JSON object as described. Do not include explanations unless requested.
'''
        return prompt

@PROMPT_REGISTRY.register()
class ComparisonCheckPrompt():
    '''
    The prompt for checking the comparison multihop question.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Question1: str, Answer1: str, Document1: str, Question2: str, Answer2: str, Document2: str, Final_question: str, Final_answer: str, qa_type: str) -> str:
        prompt = f'''You are a multi-hop QA verification system.
  ## Task
  You are given two question-answer-document triples:
  (Question1, Answer1, Doc1) and (Question2, Answer2, Doc2), plus a final multi-hop QA:
  - Final_question
  - Final_answer
  - type: "comparison"

  Your job is to **verify whether the final QA is logically valid** according to the reasoning paths and documents.

  ## Input Fields
  - Question1, Answer1, Doc1
  - Question2, Answer2, Doc2
  - Final_question, Final_answer
  - type: "inference"

  ## Output Format
  Return a JSON object:
  {{
    "valid": "true" | "false",
    "error_type": "forced_pairing" | "lacking_evidence" | "trivial_concatenation" | "other",
    "justification": "Short explanation of the issue"
  }}

  ## Definitions & Rules
  - "comparison": Final question compares a **shared attribute/dimension** (e.g., date, numeric quantity, size) between two entities derived from QA1 and QA2.

  - "forced_pairing": The two QA-doc pairs do not share a meaningful, comparable dimension — the comparison is forced or domain-incoherent.
  - "lacking_evidence": One or both compared values are not explicitly supported in the provided documents with sufficient precision to support the asserted comparison.
  - "trivial_concatenation": If the final question is formed by simply joining two or more independent facts from the given QA-document pairs into a single sentence (often using "and" or similar conjunctions), without any logical comparing beyond listing the facts.
  - "other": Other errors that you think are not included above.

  ## Examples

  ### ✓ Valid
  Input:
  Question1: When was John Beach born?
  Answer1: January 1, 1812
  Doc1: Major John Beach (January 1, 1812 - August 31, 1874) was a United States Army officer during the Black Hawk and American Civil War.

  Question2: When was Seth Gordon Persons born?
  Answer2: February 5, 1902
  Doc2: Seth Gordon Persons (February 5, 1902 - May 29, 1965) was an American Democratic politician and the 43rd Governor of Alabama.

  Final_question: Who was born first, John Beach or Seth Gordon Persons?
  Final_answer: John Beach
  type: "comparison"

  Expected Output:
  {{
    "valid": "true",
    "error_type": "No Error",
    "justification": "Both docs provide explicit birth dates (1812 vs 1902) and John Beach is earlier."
  }}

  ### ✖ Invalid (forced_pairing)
  Input:
  Question1: What was the renumbering date of the 17th Lancers?
  Answer1: April 1763
  Doc1: The regiment was renumbered the 17th Regiment of (Light) Dragoons in April 1763.

  Question2: What year was the 67th English cricket season?
  Answer2: 1763
  Doc2: The 1763 English cricket season was the 67th English cricket season.

  Final_question: Was the renumbering of the 17th Lancers in April 1763 before or during the 67th English cricket season?
  Final_answer: During
  type: "comparison"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "forced_pairing",
    "justification": "This forces a link between a military renumbering (point event) and a sports season (period) without a meaningful shared comparison dimension."
  }}

  ### ✖ Invalid (lacking_evidence)
  Input:
  Question1: What year was the Cambridge Battery completed for the 100-ton gun?
  Answer1: 1886
  Doc1: Cambridge Battery was ready in 1886.

  Question2: When did the 1886 United Kingdom general election take place?
  Answer2: July 1-27, 1886
  Doc2: The 1886 United Kingdom general election took place from 1 July to 27 July 1886.

  Final_question: Which event occurred first, the completion of the Cambridge Battery in 1886 or the United Kingdom general election led by Charles Stewart Parnell in 1886?
  Final_answer: The completion of the Cambridge Battery in 1886
  type: "comparison"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "lacking_evidence",
    "justification": "The Cambridge Battery is only dated to the year (1886), while the election records include exact dates in July. This lack of precise dating in the document makes it impossible to determine which event occurred first."
  }}

  ### ✖ Invalid: (trivial_concatenation)
  Input:
  Question1: Where was the State Normal School at Cheney located by the end of the term's first week?  
  Answer1: Pomeroy building 
  Doc1: By the end of the term's first week, the State Normal School at Cheney was located in the Pomeroy building.

  Question2: Who designed the Cheney Building?
  Answer2: H. H. Richardson  
  Doc2: The Cheney Building was designed by H. H. Richardson.

  Final_question: Which building was used for the State Normal School at Cheney by the end of the term's first week, and who designed the Cheney Building?
  Final_answer: Pomeroy building, H. H. Richardson
  type: "comparison"

  Expected Output:
  {{
    "valid": "false",
    "error_type": "trivial_concatenation",
    "justification": "Final_question simply concatenates the two questions together using 'and' without needing reasoning."
  }}

  The data need to be processed is as follows:
  Question1: {Question1}
  Answer1: {Answer1}
  Doc1: {Document1}
  Question2: {Question2}
  Answer2: {Answer2}
  Doc2: {Document2}
  Final_question: {Final_question}
  Final_answer: {Final_answer}
  type: {qa_type}
  Only return the JSON object as described. Do not include explanations unless requested.
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class RefineAnswerPrompt():
    '''
    The prompt for refining the answer.
    '''
    def __init__(self):
        pass

    def build_prompt(self, question: str, original_answer: str) -> str:
        prompt = f'''You are an AI agent tasked with cleaning and extracting concise answers from original QA pairs.
  ## Input:
  You are given a **question** and its corresponding **original answer**. Your task is to extract the most precise and concise information that directly answers the question.

  ## Processing Rules:
  1. Extract **only** the exact information requested in the question.
  2. Keep the original index numbering or order if present.
  3. **Do not** omit essential information.
  4. **Never add or infer** information not explicitly stated in the original answer.
  5. Follow strict formatting conventions:
    - Percentages: use format like `8%` (not "eight percent" or "8 percent")
    - Currency: use `$1,000` format
    - Dates: use `YYYY-MM-DD`
    - Units: include units (e.g., `5kg`, `10cm`)
  6. For answers that consist of multiple parts or are comparative in nature, multiple core components and comparative statements should be included.

  ## Output Format (JSON):
  For each input QA pair, output the following JSON object:
  {{
    "question": "<original question>",
    "original_answer": "<original answer>",
    "refined_answer": "<clean, concise, and direct answer>"
  }}

  ## Example:
  Input:
  question: What edition of the Wightman Cup was held in 1931?
  original_answer: The 1931 Wightman Cup was its 9th edition.

  Output:
  {{
  "question": "What edition of the Wightman Cup was held in 1931?",
  "original_answer": "The 1931 Wightman Cup was its 9th edition.",
  "refined_answer": "The 9th edition."
  }}

  Input:
  question: How does the percentage of individuals under age 18 living below the poverty line in Farina, Illinois compare to the statewide percentage in Illinois?
  original_answer: In Farina, Illinois, 10.7% of individuals under age 18 were living below the poverty line, which is lower than the statewide percentage in Illinois of 16.1%.

  Output:
  {{
  "question": "How does the percentage of individuals under age 18 living below the poverty line in Farina, Illinois compare to the statewide percentage in Illinois?",
  "original_answer": "In Farina, Illinois, 10.7% of individuals under age 18 were living below the poverty line, which is lower than the statewide percentage in Illinois of 16.1%.",
  "refined_answer": "10.7%, lower than the statewide percentage in Illinois of 16.1%."
  }}

  The data need to be processed is as follows:
  question: {question}
  original_answer: {original_answer}
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class MoreOptionalAnswersPrompt():
    '''
    The prompt for generating more optional answers.
    '''
    def __init__(self):
        pass

    def build_prompt(self, refined_answer: str) -> str:
        prompt = f'''You are an expert in **linguistic variation** and **data augmentation**. Your task is to generate a comprehensive list of all plausible and commonly recognized alternative expressions, formats, and aliases for a given input entity or piece of information. The goal is to create high-quality training data that captures diverse ways of referring to the same concept.

  **Key Guidelines:**

  1.  **Equivalence:** Each alternative expression must refer to *exactly the same entity or information* as the original input. Do not include broader categories, narrower sub-types, or related but distinct concepts.
  2.  **Scope of Variation:** Focus on:
      * Different **formatting conventions** (e.g., dates, numbers, units).
      * Common **abbreviations, acronyms, or initialisms**.
      * Well-known **aliases, nicknames, or shorter forms** in common usage.
      * Synonyms or rephrasing should *only* be included if they are direct, commonly accepted equivalents.
  3.  **Context-Agnosticism:** Unless the input itself implies a specific context, generate general-purpose variations. Avoid creating variations that are only valid in very niche or obscure contexts.
  4.  **Inclusion of Original:** Always include the original input as the first item in the generated list.
  5.  **Format:** Output the variations as a JSON list of strings.

  **Examples:**

  Input: 1977-01-26
  Output: ["1977-01-26", "1977 01 26", "1977.01.26", "January 26, 1977", "26 Jan 1977", "Jan 26, 1977"]

  Input: United Nations
  Output: ["United Nations", "U.N.", "UN"]

  Input: 3.14159
  Output: ["3.14159", "π", "pi", "PI"]

  Input: Doctor of Philosophy
  Output: ["Doctor of Philosophy", "Ph.D.", "PhD", "Doctorate"]

  Input: New York City
  Output: ["New York City", "NYC", "The Big Apple"]

  Input: kilogram
  Output: ["kilogram", "kg", "kilograms"]

  Input: {refined_answer}
  Please list all possible textual expressions that have the same meaning or refer to the same entity, especially in different formats (e.g., dates, names, abbreviations).
  Respond with a JSON list of strings. Do not explain.
'''
        return prompt
        
@PROMPT_REGISTRY.register()
class ReasoningPrompt():
    '''
    The prompt for reasoning.
    '''
    def __init__(self):
        pass

    def build_prompt(self, problem: str) -> str:
        prompt = f'''Please solve the following problem and return result. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details. 
  The problem is:
  {problem}
'''
        return prompt

@PROMPT_REGISTRY.register()
class ComparisonReasoningPrompt():
    '''
    The prompt for comparison question reasoning.
    '''
    def __init__(self):
        pass

    def build_prompt(self, problem: str) -> str:
        prompt = f'''Please solve the following problem and return result. 
  For comparison question, if you are unsure of the answer, please do not guess or choose randomly. Instead, return "I cannot answer this question." 
  The problem is:
  {problem}
  Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class SingleHopPrompt():
    '''
    The prompt for answer single hop question.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Document: str, Question: str) -> str:
        prompt = f'''You are given the following document that contains relevant information to help answer a question.
  Document:
  {Document}
  Question:
  {Question}
  Please answer the question using the information in the provided document. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.
  If you cannot answer the question, return "I cannot answer this question. <the reason why cannot answer this question>".
  ## For Example:
  ### ✓ Can answer:
  Document:
  \"Olive Winchester\"\n1851 in Corinna, Maine; died October 2, 1892 in Yankton, South Dakota), and Sarah A. \"\"Sadie\"\" Blackstone Winchester (born May 1, 1853 in Pownal, Maine; died February 6, 1949 in Los Angeles, California). Winchester's parents were married in Portland, Maine on February 22, 1879 in the Methodist Episcopal Church. Winchester was a relative of Oliver Fisher Winchester (born November 30, 1810 in Brookline, Massachusetts; died December 11, 1880 in New Haven, Connecticut), the manufacturer and marketer of the Winchester repeating rifle. After June 25, 1880, the Winchester family left Monson, Maine and by 1881 had relocated to Forestburg, in Sanborn
  Question:
  When was the first woman to graduate with a Bachelor of Divinity degree from Glasgow, Olive Winchester, born?
  Answer:
  1851

  Document:
  \"1996 North Indian Ocean cyclone season\"\ndue to flash flooding. Elsewhere in India, the storm killed 111 people, including 44 in Tamil Nadu where 18 boats were damaged or missing. In some areas, the rains helped end a drought. After the storm passed, the Andhra Pradesh government provided each family RS$1,000 (US$30) if their house was destroyed, and RS$100,000 (US$3,000) if they lost a family member. While the previous storm was paralleling the east Indian coastline, another disturbance formed off the west coast on June 15, also associated with the monsoon. The new area of convection persisted, developing a distinct circulation by the next day. 
  Question:
  What is the prime factorization of the number of people killed by the storm in India during the 1996 North Indian Ocean cyclone season?
  Answer:
  3 x 37

  Document:
  Seth Gordon Persons (February 5, 1902 - May 29, 1965) was an American Democratic politician who was the 43rd Governor of Alabama from 1951 to 1955.
  Question:
  Who was born first, John Beach (January 1, 1812 - August 31, 1874) or Seth Gordon Persons?
  Answer:
  John Beach

  ### ✖ Cannot answer:
  Document:
  The Rialto Bridge (Italian: Ponte di Rialto; Venetian: Ponte de Rialto) is the oldest of the four bridges spanning the Grand Canal in Venice, Italy. Connecting the sestieri (districts) of San Marco and San Polo, it has been rebuilt several times since its first construction as a pontoon bridge in the 12th century, and is now a significant tourist attractiojn the city.
  Question:
  What is the name of the famous bridge in the place where Al gran sole carico d'amore's composer worked?
  Answer:
  I cannot answer this question. I don’t know where the composer of Al gran sole carico d’amore has worked.

  Document:
  Worst (manga) Worst is a Japanese delinquent manga series written and illustrated by Hiroshi Takahashi. It has the same setting as Takahashi's previous manga \"\"Crows\"\" and \"\"QP\"\" and revolves around a group of teenage boys who fight their way through the notorious high school, \"\"Suzuran\"\". The manga was first published by Shōnen Champion in 2002. The series is currently being serialized in Japan and has been collected into twenty-five tankōbon volumes. In North America, Digital Manga Publishing has released only three volumes, with the last graphic novel released in November 2004. The series is currently on hiatus but Digital Manga
  Question:
  Which of the following was critiqued as one of the worst manga, .hack//Legend of the Twilight or the manga Worst, published in 2002?
  Answer:
  I cannot answer this question. I don't have sufficient information about "hack/Legend of the Twilight".
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class MultihopInferencePrompt():
    '''
    The prompt for answer multihop inference question.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Data: str, FinalQuestion: str) -> str:
        prompt = f'''You are an expert at solving problems. Now you need to solve a multi-hop inference problem.
  Multi-hop inference promblem: a question that requires combining information from multiple sources in a logical chain to reach an answer.

  ## For Example:
  Input:
  Question1: "What is the name of the performer of Qui de nous deux?"
  Answer1: "Matthieu Chedid"
  Supporting Document1: "'Qui de nous deux' is performed by Matthieu Chedid."
  Question2: "Who is the father of Matthieu Chedid?"
  Supporting Document2: "Matthieu Chedid is the son of Louis Chedid."
  FinalQuestion: "Who is the father of the performer of Qui de nous deux?"
  Output:
  "Louis Chedid"

  ## The Example's Logic Chain (Just to help you better understand the multihop problem. Don't output something like this):
  FinalQuestion:"Who is the father of the performer of Qui de nous deux?" -> Q:"What is the name of the performer of Qui de nous deux?" A:"Matthieu Chedid" -> Q:"Who is the father of Matthieu Chedid?" A:"Louis Chedid" -> FinalAnswer: "Louis Chedid"
  
  ## Now you are given some supporting fact to help answer a question.
  {Data}
  FinalQuestion: {FinalQuestion}
  Now you need sole the promblem and return result. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.
'''
        return prompt

@PROMPT_REGISTRY.register()
class MultihopComparisonPrompt():
    '''
    The prompt for answer multihop comparison question.
    '''
    def __init__(self):
        pass

    def build_prompt(self, Data: str, FinalQuestion: str) -> str:
        prompt = f'''You are an expert at solving problems. Now you need to solve a multi-hop comparison problem.
  Multi-hop comparison promblem: a question that requires retrieving and comparing information from multiple sources to determine a relative fact.
  
  ## For Example:
  Input:
  Question1: "When was John Beach born?"
  Answer1: "January 1, 1812"
  Supporting Document1: "Major John Beach( January 1, 1812 - August 31, 1874) was a United States Army officer during the Black Hawk and American Civil War."
  Question2: "When was Seth Gordon Persons born?"
  Answer2: "February 5, 1902"
  Supporting Document2: "Seth Gordon Persons( February 5, 1902 - May 29, 1965) was an American Democratic politician who was the 43rd Governor of Alabama from 1951 to 1955."
  FinalQuestion: "Who was born first, John Beach or Seth Gordon Persons?"
  Output:
  "John Beach"

  ## The Example's Logic Chain (Just to help you better understand the multihop problem. Don't output something like this):
  FinalQuestion: "Who was born first, John Beach or Seth Gordon Persons?" -> Q:"When was John Beach born?" A:"January 1, 1812" + Q:"When was Seth Gordon Persons born?" A:"February 5, 1902" -> FinalAnswer: "John Beach"
  
  ## Now you are given some supporting fact to help answer a question.
  {Data}
  FinalQuestion: {FinalQuestion}
  Now you need sole the promblem and return result. Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.
'''
        return prompt
    
@PROMPT_REGISTRY.register()
class EssEqPrompt():
    '''
    The prompt for llm judge.
    '''
    def __init__(self):
        pass

    def build_prompt(self, question, golden_answer, other_answer) -> str:
        prompt = f'''You are an expert evaluator. Evaluate whether the OTHER ANSWER preserves **all essential information** in the GOLDEN ANSWER, **with respect to the QUESTION**.
  # Scoring Criteria
  - **2 points** → OTHER ANSWER is fully equivalent to the GOLDEN ANSWER. Same meaning, even if reworded or paraphrased. No missing or incorrect information.
  - **1 point** → OTHER ANSWER includes ALL key information from the GOLDEN ANSWER but adds **extra non-contradictory information** that may not be strictly necessary but is still valid in context.
  - **0 points** → OTHER ANSWER is **missing** critical information from the GOLDEN ANSWER or introduces **incorrect/contradictory** information, based on the QUESTION.

  Always consider what the QUESTION is asking when judging whether information is essential.

  # ✓ Positive Examples

  ## ✓ 2 points

  - Question: What year did the war end?  
    Golden: 1848  
    Other: The year was 1848.

  - Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Obama

  - Question: When did the battle begin?  
    Golden: The battle began in 1775.  
    Other: The conflict started in the year 1775.

  - Question: Which field of Nobel Prize did Marie Curie receive?
    Golden: the Nobel Prize in Physics.
    Other: Physics.

  ## ✓ 1 point

  - Question: What is the cause of death of Mercedesz Henger's father?  
    Golden: diabetes mellitus.  
    Other: Diabetes mellitus type 2.

  - Question: When did the war end?  
    Golden: 1848  
    Other: The war ended in 1848.

  - Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Barack Obama, the 44th president of the United States.

  - Question: Where is the Eiffel Tower located?  
    Golden: The Eiffel Tower is in Paris.  
    Other: The Eiffel Tower is in Paris, the capital of France.

  ## ✖ Negative Examples (0 points)

  - Question: How much is the price?  
    Golden: 50  
    Other: 25 dollars

  - Question: When did the war end?  
    Golden: 1848  
    Other: In 1846

  - Question: Where is the Eiffel Tower located?  
    Golden: The Eiffel Tower is in Paris.  
    Other: The Eiffel Tower is in Berlin.

  - Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Barack Obama and Abraham Lincoln were presidents during the same era.

  # Output Format:
  Return ONLY JSON, no extra text.
  ```json
  {{
    "answer_reason": "reason for the score",
    "answer_score": 0/1/2
  }}
  ```
  Input:
  Question: {question}
  Golden answer: {golden_answer}
  Other answer: {other_answer}
'''
        return prompt