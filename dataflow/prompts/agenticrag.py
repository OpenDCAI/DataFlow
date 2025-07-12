import json

class AutoPromptGeneratorPrompt:
    '''
    The prompt for the AutoPromptGenerator.
    '''
    def __init__(self):
        pass

    def auto_prompt_generator_prompt(self, seed_data: str) -> str:
        system_prompt = f'''You will be given a piece of seed data, which may consist of a paragraph, dialogue, or any other form of text containing potential question-answer information.
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

        return system_prompt


class QAScorerPrompt:
    '''
    The prompt for the RAG scorer.
    '''
    def __init__(self):
        pass

    def question_quality_prompt(self) -> str:
        system_prompt = '''You are an expert question quality evaluator. Given a single question from a QA dataset, your job is to assess the **clarity and meaningfulness** of the question. Specifically, judge whether the question is clearly defined, unambiguous, and worth asking in a real-world or task-specific context.

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

        return system_prompt

    def answer_alignment_prompt(self) -> str:
        system_prompt = '''You are a response alignment evaluator. Your task is to assess whether a given answer **directly and clearly addresses the given question**.

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

        return system_prompt

    def answer_verifiability_prompt(self) -> str:
        system_prompt = '''You are an evaluator tasked with assessing how **easily verifiable** an answer is. You must determine whether the correctness of the answer can be **conveniently and unambiguously judged** — for example, whether it is fact-based, precise, and not subjective or vague.

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

        return system_prompt

    def downstream_value_prompt(self) -> str:
        system_prompt = '''You are a task relevance evaluator. Given a QA pair, assess how well this data point could **support a downstream task** such as classification, dialogue, retrieval, summarization, or knowledge grounding.

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

        return system_prompt

class AtomicTaskGeneratorPrompt:
    '''
    The prompt for the AtomicTaskGenerator.
    '''
    def __init__(self):
        pass
    
    def get_identifier_system_prompt(self) -> str:
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
    
    def get_identifier_prompt(self, input) -> str:
        prompt = f'''
        Now process this question:{input}
        '''
        return prompt
    
    def initial_conclusion_system_prompt(self) -> str:
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
    
    def initial_conclusion_prompt(self, input) -> str:
        prompt = f'''
    The document content to be processed is as follows: {input}
    '''
        return prompt
    
    def initial_question_system_prompt(self) -> str:
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
    
    def initial_question_prompt(self, identifier, conclusion, relation) -> str:
        prompt = f'''
        Data to be Processed:
        ID: {identifier}
        R: {relation}
        A: {conclusion}
        '''

        return prompt
    
    def clean_qa_system_prompt(self) -> str:
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
    
    def clean_qa_prompt(self, input) -> str:
        prompt = f'''
            The data need to be processed is as follows: {input}
        '''

        return prompt

    def llm_answer_prompt(self, input) -> str:
        prompt = f'''Please solve the following problem and return as many relevant results as possible that meet the query requirements.\n Ensure responses are as concise as possible, focusing only on key information while omitting redundant details.\n The task is:\n {input}
        '''.strip()
        
        return prompt
    
    def recall_system_prompt(self) -> str:
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
    
    def recall_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt

    def more_optional_answer_system_prompt(self) -> str:
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

    def more_optional_answer_prompt(self, answer) -> str:
        prompt = f"""
    The original answer is: {answer}
    Please list all possible textual expressions that have the same meaning or refer to the same entity, especially in different formats (e.g., dates, names, abbreviations).
    Respond with a JSON list of strings. Do not explain.

        """
        return prompt

    def golden_doc_prompt(self, golden_doc, question) -> str:
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
    
class DepthQAGeneratorPrompt:
    '''
    The prompt for the AtomicTaskGenerator.
    '''
    def __init__(self):
        pass
    
    def get_identifier_system_prompt(self) -> str:
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
    
    def get_identifier_prompt(self, input) -> str:
        prompt = f'''
        Now process this question:{input}
        '''
        return prompt
    
    def get_backward_task_prompt(self, input) -> str:
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
    
    def check_superset_system_prompt(self) -> str:
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
    
    def check_superset_prompt(self, new_id, relation, identifier) -> str:
        prompt = f'''
Given superset: {new_id}\n
Given relationship: {relation}\n
Given subset: {identifier}\n
'''
        return prompt
    
    def get_question_system_prompt(self) -> str:
        system_prompt = '''
  Please generate a question based on the content of the input identifier, a certain answer, and a certain relationship (this relationship is the relationship between the content of the file corresponding to the identifier and the given answer), such that
  The answer to this question is the input answer.
  The content of this question is determined by the content of the identifier and the content of the given relationship.
  The generated question should not involve the content of the input answer.
  Please return it in JSON format, with the key of the JSON being new_query.
'''
        return system_prompt
    
    def get_question_prompt(self, new_id, relation, identifier) -> str:
        prompt = f'''
                Certain answer: {identifier}\n
                Identifier: {new_id}\n
                Relationship: {relation}\n
'''
        return prompt
    
    def llm_answer_prompt(self, input) -> str:
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

    def recall_system_prompt(self) -> str:
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
    
    def recall_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt
    
class WidthQAGeneratorPrompt:
    '''
    The prompt for the AtomicTaskGenerator.
    '''
    def __init__(self):
        pass
    
    def merge_prompt_system_prompt(self) -> str:
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
    
    def merge_prompt_prompt(self, input) -> str:
        prompt = f'''
        Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), question (original question), and content_identifier (identifier).
'''
        return prompt
    
    def check_origin_system_prompt(self) -> str:
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
    
    def check_origin_prompt(self, input) -> str:
        prompt = f'''
    Here are the base questions to process:
    {json.dumps(input, indent=2, ensure_ascii=False)}
    Each dictionary contains: index (unique ID), complex_question (original complex question), 
    and original_questions (list of original questions).
'''
        return prompt
    
    def question_verify_system_prompt(self) -> str:
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
    
    def question_verify_prompt(self, input) -> str:
        prompt = f'''
    Please answer these research questions:
    {json.dumps(input, indent=2, ensure_ascii=False)}
'''
        return prompt
    
    def llm_answer_prompt(self, input) -> str:
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

    def recall_system_prompt(self) -> str:
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
    
    def recall_prompt(self, golden_answer, llm_answer) -> str:
        prompt = f'''
    The inputs are as follows:
    Golden Answer: {golden_answer}
    Other Answer: {llm_answer}
        '''
        return prompt