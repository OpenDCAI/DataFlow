'''
A collection of prompts for the general text operator.
'''
class PretrainGeneratorPrompt:
    
    def __init__(self):
        pass
    
    def pt_generate_prompt(self, content: str) -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template.
        """
        prompt = """
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. 
        Convert the following paragraph into a conversational format with multiple tags of "Question:" followed by "Answer:":

        You can only output as the given format:
        Question: xxx Answer: xxx
        Question: xxx Answer: xxx
        Now please covert the content below.
        {content}
        """
        return prompt.format(content=content)
    
class SupervisedFinetuneGeneratorPrompt:
    
    def __init__(self):
        pass

    def sft_generate_prompt(self, num_questions: int = 1, content: str = "") -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template.
        """
        prompt = """You are tasked with creating high-quality SFT data for LLM training. Please generate a batch of data where the only task is to create questions based on the provided context. Focus on diversity, relevance, and clarity when forming the questions. Please ensure each question is distinct and appropriately aligned with the context.
        Your Questions must meet the following requirements:
        1. You must strictly create exactly {num_questions} questions at a time.
        2. These {num_questions} questions must cover different aspects of the topic, ensuring diversity and comprehensiveness.
        3. Each question should be framed based on the provided context and should include sufficient background information. The questions should not be abrupt or vague.
        4. Please output the {num_questions} questions in the following JSON format:
        {{
            "instruction": "QUESTION",
            "output": "ANSWER"
        }}
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

        Please create {num_questions} distinct and well-formed questions based on the following context:""".format(num_questions=num_questions)
        return f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant"

import textwrap

import textwrap

class MetaPrompt:
    def __init__(self):
        self.dimensions = [
            """1. Text Structure: Evaluate the surface-level quality of the text, including spelling accuracy, grammar, vocabulary richness, and sentence structure.

Good example: "The experimental procedure was meticulously documented, with each variable clearly defined."
Bad example: "teh data was wrong and we dont no why it happen like that"

""",
            """2. Diversity and Complexity: Assess how rich and conceptually varied the content is, and whether it requires expert or deep reasoning to understand.

Good example: "This article compares Bayesian inference and frequentist approaches in statistical modeling, highlighting theoretical and practical trade-offs."
Bad example: "Dogs are pets. They bark. They are friendly."

""",
            """3. Fluency and Understandability: Evaluate whether the text flows naturally, is easy to follow, and avoids awkward or disjointed phrasing.

Good example: "Despite initial challenges, the team successfully completed the deployment by adhering to a revised strategy."
Bad example: "The problem was and then fixed by something happens deployment successful maybe."

""",
            """4. Safety: Identify whether the text contains profanities, hate speech, or excessive personally identifiable information (PII).

Good example: "The software collects anonymous usage data to improve performance."
Bad example: "You idiot, your address 123 Main St will be posted online."

""",
            """5. Educational Value: Determine whether the text provides insight, stimulates thinking, or offers meaningful learning potential.

Good example: "Understanding the principles of thermodynamics allows engineers to design more efficient engines."
Bad example: "The sky is blue. Water is wet. This is how it is."

""",
            """6. Content Accuracy and Effectiveness: Assess the truthfulness, relevance, and practical usefulness of the content.

Good example: "Newton's second law states that F = ma, which explains the relationship between force, mass, and acceleration."
Bad example: "The Earth is flat and doesn’t rotate around the Sun."

"""
        ]

        self.system_prompt_template = textwrap.dedent("""\
            You are an expert evaluator of text content. You will be given a single piece of text and must evaluate it across six specific dimensions listed below. Each dimension includes a description and two concrete examples: one high-quality ("Good example") and one low-quality ("Bad example").

{dimensions_list}

Instructions:
- Provide a clear evaluation for each of the six dimensions based on the input text.
- Each evaluation should be one short paragraph.
- Then assign an integer score from 1 to 5 for each dimension, where:
  5 = Excellent
  4 = Good
  3 = Fair
  2 = Poor
  1 = Very Poor

- Your output should end with a **separate final line** that contains a Python-style list of six integers in this format:
  [5, 4, 3, 5, 4, 5]
        """)

        self.user_prompt_template = textwrap.dedent("""\
            Please analyze and evaluate the following text:

Text:
{text}

Your output should include:
- One paragraph of analysis for each of the six quality dimensions listed above.
- A final line with your scores in this exact format:
  [score1, score2, score3, score4, score5, score6]
        """)

    def build_system_prompt(self):
        dimensions_text = "\n".join(self.dimensions)
        return self.system_prompt_template.format(dimensions_list=dimensions_text)

    def build_user_prompt(self, text):
        return self.user_prompt_template.format(text=text)


class AlpagasusPrompt:
    def __init__(self, dimension='quality'):
        self.dimension = dimension
        self.system_prompt_template = """
        We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.
        Instruction: {instruction}
        Input: {input}
        Response: {response}
        """
        self.user_prompt_template = """
        Please rate according to the {dimension} of the response to the instruction and the input. Each assistant
        receives a score on a scale of 0 to 5, where a higher score indicates a higher level of the {dimension}. Please
        first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
        """

    def build_system_prompt(self, instruction, input_text, response):
        """
        生成system prompt
        """
        return self.system_prompt_template.format(instruction=instruction, input=input_text, response=response)

    def build_user_prompt(self):
        """
        生成user prompt
        """
        return self.user_prompt_template.format(dimension=self.dimension)

class TreeinstructPrompt:
    def __init__(self):
        self.system_prompt_template = """
        You are an instruction rewriter. You need to parse a given user instruction into a TREE structure following Semantic Parsing in the natural language processing field.
        Procedure:
        step-1: Parse the old “instruction” to a TREE-1 through Semantic Parsing in the natural language processing field. 
        Count and return the number of nodes in TREE-1.
        Old instruction: “{instruction}”
        """

        self.user_prompt_template = """
        Please count and return the number of nodes in TREE-1. This number represents the complexity of the original instruction.
        Output the number in the single LAST line. You must ensure the last line is only the number of the tree, without other symbols, like ```.
        For example:
        4
        """
    
    def build_system_prompt(self, instruction):
        """
        根据给定的指令生成 system prompt
        """
        return self.system_prompt_template.format(instruction=instruction)
    
    def build_user_prompt(self):
        """
        生成 user prompt
        """
        return self.user_prompt_template

class ConsistentChatPrompt:
    def __init__(self):
        self.intent_categories = {
            "Problem Solving Interaction": [
                "From Problem Diagnosis to Solution Optimization"
            ],
            "Educational Interaction": [
                "From Broad Theory to Specific Scenarios",
                "From Basic Concepts to Cross-Domain Connections"
            ],
            "Health Consultation Interaction": [
                "From Problem Diagnosis to Solution Optimization",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Exploratory Interaction": [
                "From Time Sequence Expansion to Explore Causes and Effects",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Entertainment Interaction": [
                "From Single Perspective to Multiple Perspectives",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Simulation Interaction": [
                "From User Needs to Solutions",
                "From Broad Theory to Specific Scenarios"
            ],
            "Emotional Support Interaction": [
                "From Single Perspective to Multiple Perspectives",
                "From User Needs to Solutions"
            ],
            "Information Retrieval Interaction": [
                "From Basic Concepts to Cross-Domain Connections",
                "From Time Sequence Expansion to Explore Causes and Effects"
            ],
            "Transaction Interaction": [
                "From User Needs to Solutions",
                "From Problem Diagnosis to Solution Optimization"
            ]
        }
        self.topic_dict = {
            "Problem Solving Interaction": [
                "Technical support for computer hardware issues",
                "Home repair advice for plumbing problems",
                "Planning a budget-friendly vacation",
                "Fixing issues with internet connectivity",
                "Setting up a smart home system",
                "Solving problems with a broken washing machine",
                "Troubleshooting a malfunctioning printer",
                "How to repair a car engine",
                "Fixing a cracked phone screen",
                "Troubleshooting Wi-Fi network issues"
            ],
            "Educational Interaction": [
                "Learning a new language online",
                "Understanding the basics of physics",
                "Music theory and basic chord progressions",
                "The basics of machine learning and AI",
                "Introduction to computer programming",
                "Understanding the structure of DNA",
                "Exploring the history of the Roman Empire",
                "The principles of economics",
                "The process of photosynthesis in plants",
                "Studying the human circulatory system"
            ],
            "Health Consultation Interaction": [
                "Tips for maintaining a healthy diet",
                "Analyzing symptoms of the common cold",
                "Dealing with seasonal allergies",
                "Understanding mental health and depression",
                "Health benefits of regular exercise",
                "Managing high blood pressure",
                "Identifying signs of anxiety disorder",
                "Dealing with insomnia and sleep problems",
                "Coping with stress in the workplace",
                "Understanding the impact of smoking on health"
            ],
            "Exploratory Interaction": [
                "Exploring the concept of time travel",
                "Deep-sea exploration and underwater ecosystems",
                "Historical events that shaped the world",
                "The impact of artificial intelligence on society",
                "Exploring the mysteries of the Bermuda Triangle",
                "Investigating space exploration and Mars missions",
                "The history of human migration",
                "The future of renewable energy",
                "The impact of global warming on biodiversity",
                "Exploring the ancient pyramids of Egypt"
            ],
            "Entertainment Interaction": [
                "Creating a video game character",
                "Writing a mystery novel",
                "Designing a new board game",
                "Exploring a new fantasy world in literature",
                "The psychology behind horror movies",
                "The evolution of action films",
                "Playing a strategic card game",
                "Exploring the art of stand-up comedy",
                "How to produce an indie film",
                "Creating an engaging video game storyline"
            ],
            "Simulation Interaction": [
                "Business negotiations and decision-making",
                "Military strategy and planning simulations",
                "Simulation for emergency disaster response",
                "Flight training using simulators",
                "Healthcare simulation for medical professionals",
                "Simulating financial market crashes",
                "Simulating environmental disaster scenarios",
                "Running a simulated space mission",
                "Simulating customer service interactions",
                "Creating a disaster management simulation game"
            ],
            "Emotional Support Interaction": [
                "Coping with the death of a loved one",
                "Supporting a friend through a breakup",
                "Dealing with feelings of loneliness",
                "Coping with stress and work-life balance",
                "Managing anxiety during uncertain times",
                "Dealing with feelings of inadequacy",
                "Supporting someone going through mental health challenges",
                "Building resilience after a setback",
                "Managing anger and frustration",
                "Finding emotional support after a major life change"
            ],
            "Information Retrieval Interaction": [
                "Finding the best tech product reviews online",
                "Looking up information on the latest scientific discoveries",
                "How to find reliable health advice on the internet",
                "Searching for a vacation destination based on reviews",
                "Finding the most recent climate change data",
                "Looking for historical documents on ancient civilizations",
                "Researching news about artificial intelligence advancements",
                "Finding user reviews for a new gadget",
                "Searching for scholarly articles on quantum computing",
                "Finding government reports on public health"
            ],
            "Transaction Interaction": [
                "Booking a flight online for a vacation",
                "How to purchase concert tickets online",
                "Making an appointment with a service provider",
                "Ordering food online for delivery",
                "Purchasing a product through an e-commerce site",
                "How to buy insurance online",
                "Scheduling a medical appointment",
                "Making a donation to a charity online",
                "Buying a gift card for a friend",
                "How to apply for a mortgage loan"
            ]
        }
        
    def get_intent_categories(self):
        return self.intent_categories
    
    def get_topic_dict(self):
        return self.topic_dict
    
    def get_query_prompt(self, info_flow, topic):
        prompt = f"""
        Task Description and Rules 
        1. Generate multiple rounds of realistic user questions based on the provided topic: 
        - Based on a single core topic (provided directly by the user), generate multiple rounds of realistic user questions, comprising 6-8 turns in total. 
        - The questions should match the characteristics of real users in natural communication: sometimes simple, sometimes vague, or including contextual backgrounds, and should reflect the language style of daily communication. 
        - Note: Avoid directly including the exact expression of the input topic in the questions. Instead, abstract it with natural and conversational language in practical scenarios. 
        
        2. Dynamic Dialogue Information Flow in Conversations: Below are the relevant steps of the information flow: {info_flow}

        The dialogue style should adhere to the following requirements: 
        - Utilize natural phrasing and vivid language, avoiding overly mechanical responses. 
        - Favor shorter sentences in questions, with occasional subject omission allowed. 
        - Ensure smooth and logical transitions through lighthearted or entertaining interjections. 
        - Permit the expression of specific personality traits and individualized tones. 
        - Proactively introduce new topics when appropriate, ensuring relevance to the current theme. 
        
        The dialogue should comply with the following generation rules: 
        - For each round of dialogue, only simulate user questions without providing answers. 
        - Ensure the conversation flows naturally and reflects realistic interactive thinking. 
        - Avoid overly polished or templated content, ensuring the questions feel authentic and relatable in life scenarios. 
        
        Output Format: 
        Multi-turn Questions in JSON Format: 
        "category": "<Core Topic of the Conversation>", 
        "turns": ["<turn_1>", "<turn_2>", "<turn_3>", "..."] 
        To generate multi-turn queries with high topic consistency, please think step-by-step. 
        The input core topic for this task is: {topic}
        """
        return prompt

    def get_response_prompt(self, topic, queries):
        prompt = f"""
        Your task is to simulate a multi-turn conversation where you progressively answer a series of user questions provided under a given topic category. For each answer, focus on delivering a natural, contextually relevant, and actionable response while considering both the current question and future questions in the sequence. The goal is to ensure consistency and logical progression throughout the dialogue and to avoid unnecessary follow-up questions in the responses simultaneously. To generate multi-turn responses with high topic consistency, think step-by-step. Key Dialogue Style Requirements are as follows: 
        Content and Structure:
        1. Directly Answer the Current Question:
        - Provide a complete, useful response to the current question without posing additional questions unless they are directly relevant to future queries. 
        - If clarification or additional steps are needed, frame these as suggestions or explanations rather than questions.
        2. Be Context-Aware:
        - Always tailor each response to the current question while remaining mindful of the context provided by prior and future questions.
        - Avoid prematurely addressing future queries but create subtle links where necessary to ensure smooth progression.
        3. Clear, Action-Oriented Responses:
        - Focus on providing actionable advice, logical explanations, or troubleshooting steps rather than speculative or rhetorical remarks.
        - Avoid long or overly complex explanations; aim for clarity and efficiency.
        Tone and Style:
        1. Conversational and Supportive:
        - Use a natural, empathetic tone that simulates real-life problem-solving interactions.
        - Avoid mechanical or overly formal responses.
        2. Economical with Words:
        - Keep responses concise but informative. Minimize extraneous content while ensuring answers have enough detail to be helpful.
        3. No Unnecessary Questions:
        - Limit unnecessary questions in the responses and focus instead on providing actionable steps or solutions directly. Avoid follow-up questions that don’t align with the next user query.
        Turn-by-Turn Instructions:
        1. Answer Exclusively for the Current Question:
        - For each turn, generate an answer that directly addresses the immediate question. Avoid revisiting past details unnecessarily unless they are highly relevant.
        - While you shouldn’t anticipate or directly answer future queries, your response should create natural openings for upcoming questions if applicable.
        2. Avoid Irrelevant Follow-Up Questions:
        - If the immediate question doesn’t require clarification, frame your response as a statement or suggestion rather than a question.
        - Maintain alignment with the logical flow of dialogue to ensure each turn is coherent.
        3. Proactively Provide Scenarios or Steps:
        - Where appropriate, guide the user with specific recommendations, troubleshooting actions, or observations they can make without requiring back-and-forth clarification.
        Output Requirements:
        The output must simulate the conversation by only providing responses (one per turn) in a sequential manner. The final format must strictly adhere to valid JSON and include the required structure.
        
        The input core topic and questions-only turns for this task is: 
        core topic: {topic}
        queries:
        {', '.join([f'User query: {query}' for query in queries])}
        """
        return prompt