'''
A collection of prompts for the general text operator.
'''
class Phi4QAGeneratorPrompt:
    
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
    
class SFTGeneratorSeedPrompt:
    
    def __init__(self, custom_prompt):
        self.custom_prompt = custom_prompt
        pass

    def sft_generate_prompt(self, content: str = "") -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template,
        with optional custom instructions to enhance flexibility.
        """
        base_prompt = """You are tasked with creating high-quality SFT data for LLM training.
    Please generate one question based on the provided context, focusing on diversity, relevance, and clarity.

    Requirements:
    1. Generate exactly one distinct and well-formed question.
    2. The question must be based on the context and include enough background for clarity.
    3. Output must follow this JSON format:
    {{
        "instruction": "QUESTION",
        "output": "ANSWER"
    }}

    Examples:
    {{
        "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points.",
        "output": "Here's an HTML page with bullet points for healthy habits: <html><body><h3>Healthy Habits:</h3><ul><li>Eating a balanced diet...</li></ul></body></html>"
    }},
    {{
        "instruction": "How can we use Python to calculate the GCD (greatest common divisor) of five numbers and express each number in terms of the GCD?",
        "output": "Here's a Python function that calculates the GCD of five numbers: def find_gcd(...) ..."
    }}

    {custom_section}

    Now, based on the following context, please generate one question:
    """

        custom_section = f"Additional instruction:\n{self.custom_prompt}\n" if self.custom_prompt else ""
        full_prompt = base_prompt.format(custom_section=custom_section)
        
        return f"<|im_start|>system\n{full_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant"


import textwrap

import textwrap

class MetaPrompt:
    
    def __init__(self, dimensions):
        self.dimensions = self.format_dimensions(dimensions=dimensions)

        self.system_prompt_template = textwrap.dedent("""\
You are an expert evaluator of text content. You will be given a single piece of text and must evaluate it across six specific dimensions listed below. Each dimension includes a description and a list of concrete examples (example_list), each labeled with a quality score. Higher scores indicate better quality. Use these examples to guide your assessment.
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
        
    def format_dimensions(self, dimensions):
        formatted_list = []

        for i, item in enumerate(dimensions, 1):
            
            examples_str = "\n".join([
                f'Example (Score: {ex["score"]}):\n"{ex["text"]}"\n'
                for ex in item["example_list"]
            ])
            block = f"""\"\"\"{i}. {item["dimension_name"]}: {item["description"]}

{examples_str}\"\"\""""
            formatted_list.append(block)
        return formatted_list


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
                "Troubleshooting Wi-Fi network issues",
                "Diagnosing problems with a non-responsive remote control",
                "How to reset a frozen smartphone",
                "Dealing with an overheating laptop",
                "Replacing a broken laptop screen",
                "How to upgrade computer RAM",
                "Fixing a leaking faucet",
                "How to unclog a kitchen sink",
                "Diagnosing a noisy refrigerator",
                "How to seal window drafts",
                "Troubleshooting a non-working ceiling fan",
                "Setting up a home office on a budget",
                "Fixing a car that won’t start in cold weather",
                "How to troubleshoot GPS navigation issues",
                "Fixing problems with a garage door opener",
                "Troubleshooting smart light bulbs that won’t connect",
                "Replacing a broken door lock",
                "Fixing a noisy air conditioning unit",
                "Troubleshooting camera connectivity on a laptop",
                "How to repair a broken headphone jack",
                "Setting up a secure home Wi-Fi network",
                "Replacing a smartphone battery",
                "Installing a wall-mounted TV safely",
                "Calibrating a smart thermostat",
                "Fixing screen flickering on a monitor",
                "Diagnosing strange noises from a desktop computer",
                "Solving Bluetooth connection problems",
                "Repairing a jammed paper shredder",
                "Troubleshooting slow smartphone performance",
                "How to stop water leakage under a bathroom sink",
                "Installing weather stripping on doors",
                "Setting up parental controls on a router",
                "Fixing a dishwasher that won’t drain",
                "Repairing a damaged phone charging port",
                "Replacing a worn-out windshield wiper",
                "How to fix a garage light that keeps flickering",
                "Solving battery drain issues in electric vehicles",
                "Resetting a smart TV to factory settings",
                "Troubleshooting a wireless keyboard that won't connect",
                "How to install a backup camera in a car"
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
                "Studying the human circulatory system",
                "Learning algebra and solving equations",
                "The basics of chemistry and atomic structure",
                "Studying world geography",
                "Learning about climate change and sustainability",
                "Understanding how the internet works",
                "Intro to creative writing techniques",
                "Basics of digital photography",
                "Understanding historical timelines",
                "Learning financial literacy and budgeting",
                "Exploring different art movements",
                "Understanding gravity and Newton’s laws",
                "Learning HTML and CSS for web design",
                "Exploring the solar system",
                "Basics of environmental science",
                "Introduction to statistics",
                "Learning about the American Civil War",
                "Understanding cultural anthropology",
                "Exploring human anatomy",
                "Learning basic sign language",
                "Intro to public speaking skills",
                "Introduction to ethical philosophy",
                "Learning how to conduct scientific experiments",
                "Studying global political systems",
                "Understanding basic genetics and heredity",
                "Learning how to analyze literature",
                "Basics of entrepreneurship and starting a business",
                "Studying ancient civilizations like Mesopotamia and Egypt",
                "Introduction to psychology and behavior",
                "Basics of digital citizenship and online safety",
                "Understanding the water cycle and weather patterns",
                "Learning how to write a research paper",
                "Studying global religions and belief systems",
                "Intro to logic and critical thinking",
                "Understanding supply and demand in markets",
                "Learning spreadsheet skills (e.g., Excel or Google Sheets)",
                "Introduction to cybersecurity principles",
                "Understanding different learning styles",
                "Basics of health and nutrition science",
                "Learning how to debate effectively"
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
                "Understanding the impact of smoking on health",
                "Preventing type 2 diabetes through lifestyle changes",
                "Dealing with chronic back pain at home",
                "How to support immune health naturally",
                "Recognizing early signs of dehydration",
                "Understanding the effects of caffeine on the body",
                "Managing cholesterol through diet",
                "How to build a sustainable workout routine",
                "Mental health tips for remote workers",
                "Safe exercises for people with joint pain",
                "How to talk to a doctor about personal health concerns",
                "Advice for managing menstrual cramps",
                "Tips for healthy weight loss",
                "Understanding the role of sleep in mental wellness",
                "How to identify food intolerances",
                "Preventing common sports injuries",
                "Maintaining good posture while working",
                "Recognizing early signs of burnout",
                "How to manage asthma symptoms",
                "The importance of hydration for brain function",
                "Understanding the risks of sedentary lifestyles",
                "Managing digestive issues like bloating or IBS",
                "How to support bone health as you age",
                "Tips for quitting alcohol or reducing intake",
                "Understanding the benefits of mindfulness and meditation",
                "Recognizing signs of vitamin deficiency",
                "Safe stretching routines for flexibility",
                "How to create a balanced meal plan",
                "Managing migraines and chronic headaches",
                "Supporting eye health in the digital age",
                "Understanding how hormones affect mood and health",
                "Caring for skin during seasonal changes",
                "Understanding the basics of reproductive health",
                "Dealing with minor injuries at home (cuts, sprains)",
                "Tips for building mental resilience",
                "Creating a daily self-care routine",
                "Navigating food labels and nutrition facts",
                "Identifying signs of eating disorders",
                "How to stay active while traveling",
                "The role of gut health in overall wellness"
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
                "Exploring the ancient pyramids of Egypt",
                "Uncovering the secrets of black holes",
                "The cultural significance of ancient myths",
                "Exploring parallel universes and multiverse theories",
                "The origins and evolution of language",
                "How ancient civilizations built megastructures",
                "The search for extraterrestrial life",
                "How volcanoes have shaped Earth’s surface",
                "The psychology of dreams and their meanings",
                "The science behind natural disasters",
                "Exploring the concept of simulated reality",
                "How ancient trade routes influenced global development",
                "Exploring lost civilizations and archaeological mysteries",
                "The evolution of the internet and digital culture",
                "How pandemics have influenced human history",
                "The ethics of genetic modification",
                "Exploring the possibility of underwater cities",
                "How cultural identity evolves through migration",
                "The role of philosophy in modern science",
                "Unsolved mysteries in astrophysics",
                "Exploring ancient astronomical observatories",
                "The influence of mythologies on modern storytelling",
                "How ancient weather patterns affected human settlement",
                "Exploring the idea of colonizing other planets",
                "The rise and fall of legendary empires",
                "The possibility of time dilation in deep space travel",
                "The influence of alchemy on early science",
                "Understanding cryptids and mythological creatures",
                "Exploring the legends of Atlantis",
                "How music evolved across civilizations",
                "The significance of sacred geometry in ancient structures",
                "How ancient calendars predicted celestial events",
                "The philosophy of consciousness and existence",
                "Exploring the science behind telepathy and ESP",
                "The history of espionage and intelligence gathering",
                "How plagues transformed the course of empires",
                "The psychology behind conspiracy theories",
                "Exploring the idea of digital immortality",
                "How ancient seafaring changed the world map",
                "The role of chaos theory in understanding the universe"
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
                "Creating an engaging video game storyline",
                "Writing a screenplay for a short film",
                "Building a fantasy football team",
                "Exploring behind-the-scenes movie production",
                "Learning the basics of animation",
                "Creating your own comic book series",
                "Composing an original song",
                "Understanding character arcs in drama series",
                "Creating a YouTube channel for entertainment",
                "Developing a murder mystery dinner party game",
                "Exploring cosplay and costume design",
                "Designing the rules for a role-playing game",
                "Recording a podcast about pop culture",
                "Writing a fan fiction story",
                "Creating a music video on a budget",
                "Directing a scene with amateur actors",
                "Exploring live streaming as an entertainer",
                "Hosting an online trivia night",
                "Analyzing what makes a sitcom successful",
                "Creating viral content for social media",
                "Building a digital art portfolio for entertainment",
                "Learning how to voice act for animations or games",
                "Creating an interactive story with branching choices",
                "Reviewing and critiquing movies or TV shows",
                "Designing merchandise for a fictional brand",
                "Building a fictional world map for a fantasy series",
                "Creating theme music for a character or story",
                "Learning stage acting vs. screen acting",
                "Writing and performing a comedy skit",
                "Planning a virtual concert or talent show",
                "Designing a puzzle game with narrative elements",
                "Writing a parody song or video",
                "Hosting a fictional radio show",
                "Analyzing storytelling techniques in video games",
                "Developing an ARG (Alternate Reality Game)",
                "Creating concept art for a fantasy setting",
                "Writing dialogue for an animated series",
                "Planning a short film festival with friends",
                "Exploring sound design for entertainment media",
                "Building a fan community around fictional works"
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
                "Creating a disaster management simulation game",
                "Simulating a day in the life of a CEO",
                "Virtual reality driving test training",
                "Crisis management simulation for public relations",
                "Political campaign simulation and voter behavior",
                "Simulating ethical dilemmas in AI development",
                "Simulating the spread of infectious diseases",
                "Urban planning simulation for smart cities",
                "Simulating climate change over 100 years",
                "Training simulations for cybersecurity breaches",
                "Economic policy decision-making simulation",
                "Simulating courtroom trials and legal strategy",
                "Simulation for emergency room triage",
                "Virtual surgery practice for medical students",
                "Simulating supply chain disruptions",
                "Simulating archaeological digs and discoveries",
                "Spacewalk training in zero-gravity simulation",
                "Language learning through role-playing simulation",
                "Simulating diplomatic negotiations between countries",
                "Astronaut survival training simulation",
                "Simulating startup business pitch competitions",
                "Simulating historical battles for education",
                "Virtual restaurant management and customer flow simulation",
                "Simulating the effects of social media algorithms",
                "Driving public transportation in urban simulations",
                "Simulating a courtroom debate in a mock trial",
                "Disaster recovery planning for IT infrastructure",
                "Simulating election outcomes based on real-time data",
                "Simulation of water resource management in agriculture",
                "Creating a theme park operations simulator",
                "Simulating robotics navigation in dynamic environments",
                "Simulated coaching for sports teams",
                "Simulating ethical decision-making in journalism",
                "Simulating airport ground operations and logistics",
                "Simulating the development of a new pharmaceutical drug",
                "Simulated investment portfolio risk management",
                "Simulating refugee crisis response scenarios",
                "Virtual museum curation and exhibition planning",
                "Simulating interpersonal communication in therapy sessions",
                "Simulating AI behavior in self-driving vehicles",
                "Virtual internship simulation for workplace readiness"
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
                "Finding emotional support after a major life change",
                "Handling the emotional impact of job loss",
                "Coping with social anxiety in group settings",
                "Dealing with the fear of failure",
                "Recovering from a toxic relationship",
                "Supporting a child through emotional distress",
                "Dealing with homesickness when living abroad",
                "Finding motivation during depressive episodes",
                "Coping with a chronic illness diagnosis",
                "Navigating emotional burnout as a caregiver",
                "Overcoming feelings of rejection",
                "Learning to forgive yourself after a mistake",
                "Supporting a partner dealing with trauma",
                "Handling the emotions of being a new parent",
                "Rebuilding confidence after public embarrassment",
                "Managing expectations during major life transitions",
                "Dealing with guilt from past decisions",
                "Helping someone through a panic attack",
                "Coping with grief after a pet passes away",
                "Facing loneliness during the holiday season",
                "Balancing emotional vulnerability and self-protection",
                "Processing emotions after a traumatic event",
                "Helping teens deal with peer pressure",
                "Managing jealousy in a relationship",
                "Supporting an elderly parent with emotional needs",
                "Navigating friendship breakups with maturity",
                "Coping with fear of the future",
                "Dealing with body image issues and self-worth",
                "Handling emotional distance in long-term relationships",
                "Managing stress related to academic pressure",
                "Providing comfort to someone experiencing shame",
                "Processing mixed emotions after a big achievement",
                "Supporting someone with PTSD triggers",
                "Coping with infertility and emotional distress",
                "Rebuilding trust after betrayal",
                "Helping a loved one experiencing suicidal thoughts",
                "Dealing with emotional triggers in daily life",
                "Finding peace with an unresolved conflict",
                "Managing emotions after relocation or immigration",
                "Coping with fear of abandonment"
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
                "Finding government reports on public health",
                "Locating top-rated online courses for career development",
                "Finding official information on visa requirements",
                "Researching the latest trends in the stock market",
                "Finding statistical data for academic research",
                "Looking up real-time traffic and commute updates",
                "Finding reviews and ratings for local restaurants",
                "Searching for housing market reports in a specific city",
                "Finding information on upcoming local events",
                "Researching criminal records or public legal cases",
                "Finding comparison data on different insurance policies",
                "Searching for open-source software alternatives",
                "Looking up case studies for business or marketing",
                "Finding details on government aid programs",
                "Researching side effects of prescription medications",
                "Finding technical documentation for programming libraries",
                "Looking up airline safety records",
                "Searching for consumer complaint databases",
                "Finding educational videos on historical topics",
                "Researching the genealogy of a family name",
                "Looking up employment law information by state",
                "Finding patent information for a new invention",
                "Researching cultural practices in different countries",
                "Searching for reviews of online learning platforms",
                "Finding data on renewable energy usage by country",
                "Looking up public records on local property ownership",
                "Finding historical weather data for a location",
                "Searching for quotes and citations in classic literature",
                "Finding nutrition information for restaurant meals",
                "Researching ethical sourcing of fashion brands",
                "Looking up vehicle recall history by VIN",
                "Finding demographic data for a specific region",
                "Researching nonprofit organization transparency reports",
                "Searching for academic conference proceedings",
                "Finding ratings and reviews of mobile apps",
                "Looking up historical election results by district",
                "Finding documentation on space exploration missions",
                "Researching funding opportunities for small businesses",
                "Searching for media coverage on social justice issues",
                "Finding open data sets for machine learning training",
                "Looking up safety information on household chemicals"
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
                "How to apply for a mortgage loan",
                "Renewing a vehicle registration online",
                "Paying utility bills through a mobile app",
                "Booking a hotel room for a weekend trip",
                "Registering for an online course or certification",
                "Subscribing to a streaming service",
                "Buying event tickets with a digital wallet",
                "Applying for a credit card through a website",
                "Reserving a rental car at the airport",
                "Paying property taxes online",
                "Purchasing digital books or audiobooks",
                "Ordering groceries from an online supermarket",
                "Paying tuition fees through a university portal",
                "Signing up for a gym membership online",
                "Applying for unemployment benefits digitally",
                "Reserving a table at a restaurant using an app",
                "Buying and downloading software securely",
                "Sending money internationally via online banking",
                "Registering a domain and hosting a website",
                "Buying stocks or cryptocurrency through a trading platform",
                "Purchasing travel insurance before a trip",
                "Ordering custom clothing or merchandise online",
                "Buying a used car through an online marketplace",
                "Paying for public transportation with a mobile wallet",
                "Subscribing to a monthly subscription box service",
                "Purchasing online advertising for a small business",
                "Topping up a prepaid phone plan online",
                "Paying for freelance services via a gig platform",
                "Placing a mobile order for in-store pickup",
                "Applying for a personal loan through a fintech app",
                "Booking a guided tour or local experience online",
                "Paying entry fees for a virtual event or webinar",
                "Setting up automatic payments for monthly bills",
                "Buying furniture or home goods with financing options",
                "Purchasing digital game content or in-app items",
                "Contributing to a crowdfunding campaign",
                "Paying for parking through a mobile parking app",
                "Ordering prescription medication online",
                "Reserving coworking space for remote work",
                "Paying for tutoring or online lessons"
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
    

class CondorPrompt:
    def __init__(self):
        self.tag = {
            "Marriage and Relationships": {
                "Dating and Friendship": ["Dating Platforms", "Dating Tips", "Dating Events"],
                "Marriage Management": ["Marital Relationships", "Marriage Law", "Marriage Counseling"],
                "Wedding Planning": ["Wedding Planning", "Wedding Photography", "Wedding Venues"],
                "Relationship Psychology": ["Relationship Psychology", "Communication Skills in Relationships", "Relationship Maintenance"],
                "Emotional Counseling": ["Solving Emotional Issues", "Emotional Repair", "Emotional Growth"],
                "Pre-Marriage Education": ["Pre-Marriage Preparation", "Pre-Marriage Psychology", "Pre-Marriage Legal Knowledge"]
            },
            "Entertainment Gossip": {
                "Celebrity News": ["Celebrity News", "Celebrity Interviews", "Celebrity Charity Events"],
                "Variety Shows": ["Show Recommendations", "Behind the Scenes", "Show Interaction"],
                "Film and TV Reviews": ["Movie Reviews", "TV Series Reviews", "Critics’ Opinions"],
                "Entertainment News": ["Latest Entertainment News", "Entertainment Events", "Exclusive Interviews"],
                "Fan Culture": ["Fan Activities", "Fan Support", "Fan Interactions"],
                "Gossip": ["Celebrity Gossip", "Entertainment Industry Secrets", "Gossip Chasing"]
            },
            "Artificial Intelligence": {
                "Machine Learning": ["Algorithm Principles", "Application Cases", "Learning Resources"],
                "Deep Learning": ["Neural Networks", "Deep Learning Frameworks", "Deep Learning Applications"],
                "Natural Language Processing": ["Language Models", "Text Analysis", "Dialogue Systems"],
                "Computer Vision": ["Image Recognition", "Video Processing", "Vision Algorithms"],
                "Intelligent Robotics": ["Robotics Technology", "Service Robots", "Industrial Robots"],
                "Autonomous Driving": ["Autonomous Driving Technology", "Autonomous Driving Regulations", "Autonomous Driving Testing"]
            },
            "Healthcare": {
                "Disease Prevention and Treatment": ["Common Diseases", "Preventive Measures", "Disease Treatment"],
                "Health and Wellness": ["Dietary Wellness", "Exercise Wellness", "Traditional Chinese Medicine Wellness"],
                "Psychological Counseling": ["Mental Health Issues", "Psychological Therapy", "Psychological Adjustment"],
                "Medical Technology": ["Medical Equipment", "Medical Technology", "Medical Innovation"],
                "Health Insurance": ["Types of Insurance", "Insurance Choices", "Insurance Claims"],
                "Fitness": ["Fitness Methods", "Fitness Equipment", "Fitness Diet"]
            },
            "Pets": {
                "Pet Care": ["Daily Pet Care", "Pet Nutrition", "Pet Behavior"],
                "Pet Medical Care": ["Pet Diseases", "Pet First Aid", "Pet Hospitals"],
                "Pet Training": ["Basic Training", "Behavior Correction", "Training Techniques"],
                "Pet Supplies": ["Toys", "Food", "Care Products"],
                "Pet Adoption": ["Adoption Procedures", "Adoption Conditions", "Adoption Events"],
                "Pet Activities": ["Pet Competitions", "Pet Gatherings", "Pet Festivals"]
            },
            "Environment": {
                "Environmental Protection": ["Ecological Protection", "Pollution Control", "Environmental Monitoring"],
                "Sustainable Development": ["Green Energy", "Circular Economy", "Ecological Agriculture"],
                "Energy Conservation and Emission Reduction": ["Energy-Saving Technology", "Emission Reduction Policies", "Low-Carbon Life"],
                "Waste Sorting": ["Sorting Standards", "Sorting Methods", "Recycling"],
                "Environmental Policies": ["Policy Regulations", "Policy Interpretation", "Policy Impact"],
                "Green Living": ["Green Consumption", "Green Travel", "Green Buildings"]
            },
            "Technology": {
                "Internet": ["Network Technology", "Cybersecurity", "Online Services"],
                "5G Communication": ["5G Technology", "5G Applications", "5G Devices"],
                "Blockchain": ["Blockchain Principles", "Blockchain Applications", "Digital Currency"],
                "Artificial Intelligence": ["AI Technology", "AI Ethics", "AI Industry Applications"],
                "Aerospace": ["Aerospace Technology", "Aircraft", "Space Exploration"],
                "New Energy": ["Solar Energy", "Wind Energy", "New Energy Vehicles", "Energy Storage"]
            },
            "Education and Training": {
                "Preschool Education": ["Choosing Kindergartens", "Early Childhood Education", "Preschool Education Policies"],
                "K12 Education": ["Primary Education", "Secondary Education", "Family Education Guidance"],
                "Higher Education": ["University Major Selection", "Graduate Education", "Higher Education Policies"],
                "Vocational Training": ["Vocational Skills Training", "Professional Certifications", "Career Development Planning"],
                "Online Education": ["Online Course Recommendations", "Distance Education", "Online Learning Tips"],
                "Study Abroad and Immigration": ["Study Abroad Consultation", "Immigration Policies", "Overseas Living Guide"]
            },
            "Career Development": {
                "Career Planning": ["Career Positioning", "Career Development Paths", "Career Transition Guidance"],
                "Job Search Skills": ["Resume Writing", "Interview Skills", "Job Search Channels"],
                "Career Advancement": ["Promotion Strategies", "Workplace Performance", "Leadership Development"],
                "Interpersonal Relationships": ["Colleague Interaction", "Workplace Communication", "Workplace Etiquette"],
                "Entrepreneurship Guidance": ["Entrepreneurship Plans", "Entrepreneurship Resources", "Entrepreneurship Risk Management"],
                "Team Management": ["Team Building", "Team Collaboration", "Team Performance Management"]
            },
            "Finance and Investment": {
                "Stocks": ["Stock Market Analysis", "Stock Investment Strategies", "Stock Research"],
                "Funds": ["Fund Selection", "Systematic Investment Plans", "Fund Risk Management"],
                "Futures": ["Futures Market", "Futures Trading Skills", "Futures Risk Control"],
                "Foreign Exchange": ["Forex Trading", "Forex Market Analysis", "Forex Risk Management"],
                "Insurance": ["Insurance Product Selection", "Insurance Planning", "Insurance Claims"],
                "Financial Planning": ["Personal Finance", "Asset Allocation", "Retirement Planning"]
            },
            "Real Estate and Home Living": {
                "Real Estate Market": ["Market Trends", "Property Price Analysis", "Real Estate Policy Interpretation"],
                "Home Buying Guide": ["Home Selection Tips", "Home Buying Process", "Mortgage Application"],
                "Interior Design": ["Decorating Styles", "Decorating Materials", "Decorating Budget"],
                "Home Living": ["Home Arrangement", "Home Maintenance", "Smart Homes"],
                "Real Estate Policies": ["Policy Updates", "Policy Interpretation", "Policy Impact"],
                "Rental Market": ["Rental Process", "Rental Agreements", "Rental Tips"]
            },
            "Travel and Adventure": {
                "Domestic Travel": ["Destination Recommendations", "Domestic Travel Guides", "Travel Safety"],
                "International Travel": ["Visa Applications", "International Travel Guides", "Cultural Adaptation"],
                "Outdoor Adventures": ["Hiking", "Mountain Climbing", "Wilderness Survival Skills"],
                "Travel Guides": ["Travel Planning", "Travel Budget", "Travel Packing Lists"],
                "Travel Equipment": ["Backpack Selection", "Outdoor Gear", "Travel Essentials"],
                "Travel Photography": ["Photography Tips", "Travel Photography Works", "Photography Equipment Recommendations"]
            },
            "Food and Cooking": {
                "Food Recommendations": ["Local Delicacies", "Food Rankings", "Restaurant Recommendations"],
                "Cooking Skills": ["Basic Cooking", "Creative Cooking", "Cooking Tool Usage"],
                "Ingredient Selection": ["Ingredient Selection Tips", "Seasonal Ingredients", "Organic Ingredients"],
                "Food Culture": ["Food Culture", "Local Food Customs", "Dietary Health"],
                "Healthy Eating": ["Balanced Nutrition", "Healthy Recipes", "Dietary Wellness"],
                "Baking and Desserts": ["Dessert Making", "Baking Skills", "Dessert Ingredients"]
            },
            "Culture and Arts": {
                "Literature": ["Literary Works", "Literary Criticism", "Creative Writing Skills"],
                "Music": ["Music Styles", "Music Production", "Music Appreciation"],
                "Painting": ["Painting Techniques", "Painting Schools", "Painting Appreciation"],
                "Sculpture": ["Sculpture Art", "Sculpture Creation", "Sculpture Materials"],
                "Theater": ["Theater Performance", "Theater Creation", "Theater History"],
                "Film": ["Film Recommendations", "Film Reviews", "Film Production"]
            },
            "Sports and Fitness": {
                "Sports Events": ["Event Broadcasts", "Event Analysis", "Event History"],
                "Fitness Methods": ["Fitness Tutorials", "Fitness Plans", "Fitness Diet"],
                "Sports Equipment": ["Equipment Recommendations", "Equipment Usage", "Equipment Maintenance"],
                "Sports Celebrities": ["Celebrity Introductions", "Celebrity Interviews", "Celebrity Events"],
                "Sports Policies": ["Policy Interpretation", "Policy Impact", "Policy Updates"],
                "Sports Industry": ["Industry Trends", "Industry Investment", "Industry Cases"]
            },
            "Military and National Defense": {
                "Military News": ["News Reports", "News Analysis", "Military Updates"],
                "Defense Technology": ["Technology Advancements", "Technology Applications", "Innovative Technologies"],
                "Weapons and Equipment": ["Equipment Introduction", "Equipment Comparison", "Equipment Maintenance"],
                "Military History": ["Historical Events", "Historical Battles", "Historical Figures"],
                "Military Service System": ["Service Regulations", "Enlistment Process", "Veterans' Policies"],
                "National Security": ["Security Policies", "Security Education", "Security Awareness"]
            },
            "Social Welfare": {
                "Charity Donations": ["Donation Channels", "Donation Impact", "Donation Stories"],
                "Volunteer Services": ["Service Projects", "Service Training", "Volunteer Stories"],
                "Public Welfare Activities": ["Activity Organization", "Activity Participation", "Activity Impact"],
                "Public Welfare Organizations": ["Organization Introductions", "Organization Activities", "Organization Cooperation"],
                "Social Assistance": ["Assistance Targets", "Assistance Methods", "Assistance Policies"],
                "Spreading Love": ["Spreading Methods", "Spreading Activities", "Spreading Impact"]
            },
            "Automotive and Transportation": {
                "Automotive News": ["New Car Releases", "Car Reviews", "Automotive Trends"],
                "Driving Skills": ["Safe Driving", "Fuel-Efficient Driving", "Driver Training"],
                "Vehicle Maintenance": ["Routine Maintenance", "Fault Diagnosis", "Repair Services"],
                "Traffic Laws": ["Law Interpretation", "Safety Education", "Law Updates"],
                "New Energy Vehicles": ["Technical Features", "Market Dynamics", "Policy Support"],
                "Smart Transportation": ["Technology Applications", "Smart Systems", "Future Trends"]
            },
            "E-commerce": {
                "Online Shopping": ["Shopping Guides", "User Reviews", "Promotions"],
                "E-commerce Operations": ["Operations Management", "Market Analysis", "Customer Service"],
                "Cross-border E-commerce": ["International Logistics", "Tariff Policies", "Market Analysis"],
                "E-commerce Policies": ["Policy Interpretation", "Policy Impact", "Compliance Operations"],
                "E-commerce Marketing": ["Marketing Strategies", "Advertising Placement", "User Analysis"],
                "E-commerce Logistics": ["Logistics Delivery", "Inventory Management", "Logistics Technology"]
            },
            "Gaming and Animation": {
                "Online Games": ["Popular Games", "Game Reviews", "Gaming Communities"],
                "Single-player Games": ["Classic Games", "Game Guides", "Game Recommendations"],
                "Animation Works": ["Popular Anime", "Anime Characters", "Anime Production"],
                "Game Guides": ["Guide Sharing", "Skill Exchange", "Guide Videos"],
                "Animation Industry": ["Industry Trends", "Market Analysis", "Industry Policies"],
                "Game Merchandise": ["Merchandise Products", "Collecting Guides", "Merchandise Events"]
            },
            "Infant and Child Education": {
                "Early Education": ["Educational Philosophy", "Educational Methods", "Educational Toys"],
                "Maternal and Infant Care": ["Care Knowledge", "Care Skills", "Care Products"],
                "Child Psychology": ["Psychological Development", "Emotion Management", "Psychological Counseling"],
                "Parent-child Relationship": ["Parent-child Activities", "Parent-child Communication", "Parent-child Education"],
                "Baby Products": ["Product Selection", "Safety Standards", "Product Recommendations"],
                "Child Health": ["Healthy Growth", "Nutritional Diet", "Disease Prevention"]
            },
            "Senior Life": {
                "Elderly Care Policies": ["Policy Interpretation", "Policy Consultation", "Policy Implementation"],
                "Senior Health": ["Health Checkups", "Disease Prevention", "Healthy Eating"],
                "Senior Activities": ["Cultural Activities", "Sports Activities", "Social Activities"],
                "Senior Psychology": ["Psychological Adjustment", "Psychological Health", "Psychological Support"],
                "Elderly Care Institutions": ["Institution Selection", "Service Quality", "Institution Evaluation"],
                "Senior Products": ["Assistance Products", "Health Products", "Living Products"]
            },
            "Psychological Counseling": {
                "Mental Health": ["Mental Maintenance", "Mental Problem Prevention", "Mental Health Education"],
                "Psychological Disorders": ["Disorder Identification", "Disorder Treatment", "Disorder Management"],
                "Counseling Skills": ["Counseling Methods", "Communication Skills", "Case Studies"],
                "Psychological Tests": ["Test Types", "Test Applications", "Test Interpretation"],
                "Psychological Research": ["Research Trends", "Research Methods", "Research Results"],
                "Psychological Guidance": ["Guidance Strategies", "Guidance Cases", "Guidance Resources"]
            },
            "Religion and Belief": {
                "Religious Culture": ["Cultural Traditions", "Cultural Festivals", "Cultural Influence"],
                "Religious History": ["Historical Development", "Key Events", "Historical Figures"],
                "Religious Art": ["Art Forms", "Art Works", "Art Value"],
                "Religious Policies": ["Policy Regulations", "Policy Interpretation", "Policy Impact"],
                "Religious Activities": ["Activity Organization", "Activity Participation", "Activity Significance"],
                "Faith Discussions": ["Meaning of Faith", "Faith Conflicts", "Faith Diversity"]
            },
            "Agriculture and Rural Development": {
                "Agricultural Technology": ["Technology Applications", "Technological Innovation", "Technology Promotion"],
                "Rural Development": ["Development Planning", "Development Models", "Development Cases"],
                "Farmer Life": ["Life Improvement", "Quality of Life", "Living Customs"],
                "Agricultural Products Market": ["Market Analysis", "Market Trends", "Market Transactions"],
                "Agricultural Policies": ["Policy Support", "Policy Interpretation", "Policy Implementation"],
                "Rural Tourism": ["Tourism Development", "Tourism Projects", "Tourism Experience"]
            },
            "Urban Planning": {
                "Urban Planning": ["Planning Philosophy", "Planning Methods", "Planning Cases"],
                "Urban Design": ["Design Philosophy", "Design Elements", "Design Practice"],
                "Infrastructure Development": ["Development Planning", "Development Management", "Development Technology"],
                "Urban Transportation": ["Transportation Planning", "Transportation Management", "Transportation Tools"],
                "Urban Greening": ["Greening Layout", "Greening Technology", "Greening Effects"],
                "Protection of Historic Cities": ["Protection Policies", "Protection Measures", "Protection Cases"]
            },
            "Laws and Regulations": {
                "Civil Law": ["General Principles", "Property Law", "Contract Law"],
                "Criminal Law": ["General Principles", "Types of Crimes", "Punishment Systems"],
                "Administrative Law": ["Administrative Regulations", "Administrative Litigation", "Administrative Reconsideration"],
                "Economic Law": ["Corporate Law", "Tax Law", "Intellectual Property Law"],
                "International Law": ["Public International Law", "Private International Law", "International Trade Law"],
                "Legal Consultation": ["Consultation Services", "Legal Aid", "Legal Education"]
            },
            "Art": {
                "Painting": ["Painting Techniques", "Painting Styles", "Painting Works"],
                "Sculpture": ["Sculpture Materials", "Sculpture Styles", "Sculpture Creation"],
                "Design": ["Design Philosophy", "Design Methods", "Design Works"],
                "Photography": ["Photography Techniques", "Photography Themes", "Photography Works"],
                "Calligraphy": ["Calligraphy Art", "Calligraphy Styles", "Calligraphy Works"],
                "Handicrafts": ["Craft Making", "Craft Materials", "Craft Culture"]
            },
            "Marketing": {
                "Market Research": ["Research Methods", "Research Tools", "Research Reports"],
                "Marketing Strategies": ["Strategy Formulation", "Strategy Execution", "Strategy Evaluation"],
                "Brand Management": ["Brand Positioning", "Brand Promotion", "Brand Maintenance"],
                "Advertising": ["Creative Advertising", "Advertising Media", "Advertising Effectiveness"],
                "Public Relations": ["Event Planning", "Event Execution", "Event Evaluation"],
                "Channel Development": ["Channel Expansion", "Channel Management", "Channel Optimization"]
            },
            "Astronomy and Geography": {
                "Astronomy": ["Astronomical Observations", "Astronomical Phenomena", "Astronomical Research"],
                "Geography": ["Geographical Knowledge", "Geographical Exploration", "Geographical Education"],
                "Geology": ["Geological Structure", "Geological Survey", "Geological Protection"],
                "Meteorology": ["Weather Forecasting", "Weather Disasters", "Weather Services"],
                "Space Exploration": ["Space Exploration", "Interstellar Travel", "Extraterrestrial Life"],
                "Geographical Information Systems": ["GIS Technology", "GIS Applications", "GIS Development"]
            },
            "Education and Exams": {
                "College Entrance Exam Coaching": ["Preparation Strategies", "Practice Tests", "Exam Policy Interpretation"],
                "Graduate School Entrance Exam Coaching": ["Preparation Planning", "Specialty Coaching", "Psychological Adjustment"],
                "Civil Service Exams": ["Exam Techniques", "Essay Writing Guidance", "Interview Preparation"],
                "Teaching Qualification Exams": ["Exam Process", "Interview Skills", "Teaching Ability Improvement"],
                "Foreign Language Exams": ["CET-4/CET-6", "IELTS/TOEFL", "Foreign Language Speaking Training"],
                "Professional Qualification Exams": ["Exam Subjects", "Career Development", "Qualification Certification"]
            },
            "Cybersecurity": {
                "Cybersecurity Protection": ["Protection Measures", "Security Tools", "Protection Strategies"],
                "Hacker Attack and Defense": ["Attack and Defense Drills", "Security Vulnerabilities", "Hacking Techniques"],
                "Data Encryption": ["Encryption Technology", "Data Protection", "Encryption Strategies"],
                "Information Leak Prevention": ["Leakage Risks", "Prevention Measures", "Emergency Response"],
                "Cybersecurity Policies": ["Policy Interpretation", "Regulations and Standards", "Policy Updates"],
                "Cybersecurity Incidents": ["Incident Analysis", "Incident Tracking", "Incident Prevention"]
            },
            "Fashion and Trends": {
                "Clothing Matching": ["Everyday Outfits", "Dressing for Occasions", "Fashion Trends"],
                "Beauty and Skincare": ["Skincare Knowledge", "Makeup Skills", "Beauty Products"],
                "Fashion Accessories": ["Jewelry Matching", "Accessory Selection", "Trendy Accessories"],
                "Trend Analysis": ["Fashion Week", "Trend Analysis", "Trend Forecasting"],
                "Fashion Bloggers": ["Blogger Recommendations", "Blogger Styles", "Blogger Influence"],
                "Fashion Brands": ["Brand Stories", "Brand Series", "Brand Events"]
            },
            "Mental Health": {
                "Emotion Management": ["Emotion Recognition", "Emotion Regulation", "Emotion Expression"],
                "Stress Management": ["Stress Sources", "Stress Relief Techniques", "Stress Management"],
                "Interpersonal Relationships": ["Communication Skills", "Conflict Resolution", "Social Skills"],
                "Self-Awareness": ["Self-Exploration", "Self-Evaluation", "Personal Growth"],
                "Psychological Adjustment": ["Adjustment Methods", "Psychological Balance", "Psychological Resilience"],
                "Psychological Disorder Prevention": ["Disorder Knowledge", "Prevention Measures", "Health Promotion"]
            },
            "Agricultural Technology": {
                "Smart Agriculture": ["Smart Technology", "Precision Agriculture", "Agricultural Big Data"],
                "Agricultural Mechanization": ["Mechanization Applications", "Technological Innovation", "Mechanization Maintenance"],
                "Agricultural Product Processing": ["Processing Technology", "Product Innovation", "Quality Control"],
                "Agricultural Innovation": ["Innovation Cases", "Innovation Policies", "Innovation-Driven Development"],
                "Agricultural Policies": ["Policy Support", "Policy Interpretation", "Policy Implementation"],
                "Agricultural Market Analysis": ["Market Trends", "Demand Analysis", "Price Fluctuations"]
            },
            "Digital Products": {
                "Smartphone Reviews": ["Performance Testing", "User Experience", "New Releases"],
                "Computer Hardware": ["Hardware Configuration", "Hardware Upgrades", "Hardware Maintenance"],
                "Digital Cameras": ["Camera Selection", "Photography Tips", "Camera Maintenance"],
                "Wearable Devices": ["Device Functions", "Health Monitoring", "Smart Interactions"],
                "Routers": ["Router Setup", "Signal Optimization", "Network Security"],
                "Digital Accessories": ["Accessory Selection", "Device Protection", "Accessory Recommendations"]
            },
            "Home Decoration": {
                "Decoration Styles": ["Modern Minimalism", "Classical Chinese Style", "Luxurious European Style"],
                "Decoration Materials": ["Material Selection", "Material Environmental Protection", "Material Costs"],
                "Interior Design": ["Space Planning", "Furniture Selection", "Color Matching"],
                "Soft Decoration": ["Curtain Selection", "Bedding Matching", "Decorative Paintings"],
                "Feng Shui": ["Feng Shui Layout", "Feng Shui Taboos", "Feng Shui Improvements"],
                "Renovation Construction": ["Construction Process", "Construction Supervision", "Construction Safety"]
            },
            "History and Culture": {
                "Chinese History": ["Ancient History", "Modern History", "History Education"],
                "World History": ["Origins of Civilization", "Historical Events", "International Relations"],
                "Archaeological Discoveries": ["Site Excavation", "Cultural Relic Protection", "Archaeological Techniques"],
                "Historical Figures": ["Biographies", "Character Evaluations", "Historical Impact"],
                "Cultural Heritage": ["Heritage Protection", "Heritage Value", "Heritage Inheritance"],
                "Historical Research": ["Research Methods", "Academic Achievements", "Research Trends"]
            },
            "Travel Guides": {
                "Independent Travel Guides": ["Destination Recommendations", "Itinerary Planning", "Accommodation Selection"],
                "Group Travel Guides": ["Tour Agency Selection", "Group Activities", "Group Travel Advantages"],
                "Tourism Route Planning": ["Route Design", "Special Routes", "Theme Travel"],
                "Money-Saving Travel Tips": ["Budget Planning", "Spending Guides", "Discount Information"],
                "Travel Safety": ["Safety Tips", "Emergency Handling", "Insurance Selection"],
                "Travel Visas": ["Visa Applications", "Visa Policies", "Visa Documentation"]
            },
            "Food Sharing": {
                "Recipe Sharing": ["Recipe Sharing", "Cooking Skills", "Ingredient Selection"],
                "Food Recommendations": ["Special Dishes", "Local Snacks", "Restaurant Recommendations"],
                "Food Exploration": ["Exploration Guides", "Shop Reviews", "Food Maps"],
                "Food Photography": ["Photography Skills", "Food Presentation", "Visual Display"],
                "Food Reviews": ["Dish Reviews", "Restaurant Reviews", "Ingredient Reviews"],
                "Food Competitions": ["Competition Information", "Participation Guidelines", "Award-Winning Works"]
            },
            "Film and Entertainment": {
                "Movie Recommendations": ["New Movie Alerts", "Classic Movies", "Movie Rankings"],
                "TV Series Reviews": ["Popular Drama Reviews", "Series Recommendations", "Plot Analysis"],
                "Variety Show Reviews": ["Program Highlights", "Guest Performances", "Program Creativity"],
                "Online Series": ["Popular Online Series", "Online Series Production", "Online Series Trends"],
                "Short Videos": ["Short Video Creation", "Short Video Platforms", "Short Video Marketing"],
                "Film Production": ["Production Process", "Behind the Scenes", "Production Techniques"]
            },
            "Sports Activities": {
                "Ball Sports": ["Football", "Basketball", "Volleyball"],
                "Track and Field": ["Running", "Long Jump", "Throwing"],
                "Water Sports": ["Swimming", "Rowing", "Surfing"],
                "Winter Sports": ["Skiing", "Ice Skating", "Sledding"],
                "Extreme Sports": ["Rock Climbing", "Skydiving", "Extreme Cycling"],
                "Sports Events": ["International Events", "Domestic Events", "Local Events"]
            },
            "Entrepreneurship and Investment": {
                "Entrepreneurship Guidance": ["Entrepreneurship Plans", "Market Analysis", "Entrepreneurship Mindset"],
                "Investment and Finance": ["Investment Strategies", "Asset Management", "Risk Control"],
                "Entrepreneurship Policies": ["Policy Interpretation", "Policy Support", "Policy Utilization"],
                "Entrepreneurship Cases": ["Success Stories", "Lessons Learned", "Case Analysis"],
                "Venture Capital": ["Investment Opportunities", "Investment Evaluation", "Investment Negotiation"],
                "Entrepreneurship Financing": ["Financing Channels", "Financing Strategies", "Financing Agreements"]
            },
            "Music and Dance": {
                "Music Appreciation": ["Music Styles", "Music Works", "Musicians"],
                "Instrumental Performance": ["Instrument Selection", "Performance Techniques", "Instrument Maintenance"],
                "Dance Performance": ["Dance Types", "Performance Techniques", "Performance Opportunities"],
                "Music Production": ["Music Creation", "Music Recording", "Music Publishing"],
                "Music Education": ["Education Methods", "Educational Resources", "Education Policies"],
                "Dance Choreography": ["Choreography Techniques", "Choreography Creativity", "Choreography Practice"]
            },
            "National Defense and Military": {
                "Military Strategy": ["Strategy Analysis", "Strategy Planning", "Strategy Implementation"],
                "Military Training": ["Basic Training", "Tactical Training", "Special Forces Training"],
                "Weapons Development": ["Equipment Introduction", "Research and Development Updates", "Technological Innovation"],
                "Military History": ["Historical Battles", "Historical Figures", "Historical Events"],
                "National Defense Education": ["Educational Content", "Educational Methods", "Educational Significance"],
                "Military Exercises": ["Exercise Types", "Exercise Scale", "Exercise Objectives"]
            }
        }
    
    def get_question_prompt(self, theme, domain):
        """
        Generates the formatted prompt for LLM input based on the theme and domain.

        Parameters:
        theme (str): The main theme of the questions.
        domain (str): The domain under the given theme.

        Returns:
        str: The formatted prompt for generating questions.
        """
        prompt = f"""
Now we need to create high-quality SFT data for LLM training, so we need you to produce a batch of such data. You only
need to create Questions. I will give you a theme for SFT data Questions. You need to create three
Questions of different difficulty levels based on this new theme.\\
Your Questions must meet the following requirements:\\
1. You must strictly create only three Questions at a time. These three Questions must be in the domain of {domain}
and the Questions should align with the given theme of {theme}.\\
2. The Questions you create must have context and sufficient information; they should not be abrupt and directly ask the
question.\\
3. Your reply must strictly follow the format below. Your Questions need to be included between [Question Start] and
[Question End], and the difficulty level should be indicated at the beginning, as in the following format:\\

[Easy][Question Start]Question[Question End]

[Medium][Question Start]Question[Question End]

[Hard][Question Start]Question[Question End]

4. Your Questions of different difficulty levels should be distinct and actually reflect the different levels of difficulty.\\
\quad \\

Now it's your turn. Please provide the three Questions of different difficulty levels you created about the theme of {theme} for {domain}, according to the requirements.
"""
        return prompt
    
    def create_critique_prompt(self, question, answer):
        dialogue = [question, answer]
        base_critique_prompt = f"""
There is now a user’s question and a model’s response. You need to write a critique for this response, pointing out the
strengths and weaknesses of the model’s answer to help the model improve its response.

Your critique must strictly adhere to the following format:

[Critique Start]

[Strength Start]Strength[Strength End]

[Weakness Start]Weakness[Weakness End]

[Suggestion Start]Suggestion[Suggestion End]

[Critique End]

Here is the user’s question and the model’s response: {dialogue}

Now it’s your turn. Please provide your Critique as required:
        """
        return base_critique_prompt.format(dialogue=dialogue)

    def create_refine_prompt(self, question, answer, critique):
        base_refine_prompt = """
Now there is a user's question, a model's answer, and the user's feedback. Please help modify the model's answer based on the user's feedback to make it better.
Your improved answer must strictly adhere to the following format:

[Improved Answer Start]Your answer[Improved Answer End]

Below is the user's question, the model's answer, and the feedback:
[Question Start]{question}[Question End]
[Answer Start]{answer}[Answer End]
[Feedback Start]{critique}[Feedback End]

Now it's your turn, please provide your improved answer as required:
        """
        return base_refine_prompt.format(question=question, answer=answer, critique=critique)

class LanguageFilterPrompt:
    
    def __init__(self):
        pass
    
    def language_filter_prompt(self, text):
        prompt='''You are a language identification expert. Your task is to identify the language of the given text input.

        Follow these rules:You are a language identification expert. Your task is to identify the language of the given text input.

    - Respond with the ISO 639-1 two-letter language code (e.g., "en", "fr", "zh", "ar").
        - If the text contains multiple languages, identify the dominant one.
        - If the language is not identifiable, respond with "Unknown".
        - Do not translate or explain. Output only the language name.

        Here are some examples:

        Example 1:
        Text: "Hello, how are you?"
        Language: en

        Example 2:
        Text: "Je suis très heureux de vous rencontrer."
        Language: fr

        Example 3:
        Text: "これは日本語の文です。"
        Language: ja

        Example 4:
        Text: "¿Dónde está la estación de tren?"
        Language: es

        Example 5:
        Text: "مرحبا، كيف حالك؟"
        Language: ar

        Example 6:
        Text: "Guten Morgen! Wie geht's dir?"
        Language: de

        Example 7:
        Text: "你好，我是一个程序员。"
        Language: zh

        Example 8:
        Text: "Привет, как дела?"
        Language: ru

        Now, identify the language of the following text:

        Text: "{text}"
        Language:
        '''
        return prompt.format(text=text)