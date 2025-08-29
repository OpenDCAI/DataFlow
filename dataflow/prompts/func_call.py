'''
Prompts for Bottom-Up Then Top-dOwN (BUTTON) multi-turn dialogue generating pipeline.
'''

class FuncCallPrompt:
    def __init__(self):
        pass
    
    def extract_scenario_prompt(self, conversation):
        prompt = """
        Please analyze the conversation below between a user and an
        assistant bot and identify the general life scenario it
        represents. Provide a concise overview of the scenario type,
        such as 'booking flights' or 'ordering meals'. Avoid
        mentioning specific details like numbers or items. Your
        response should be a description of the scenario without
        additional commentary, and should not exceed 10 words.
        Conversation:
        {conversation}
        Concise Overview of the Scenario:
        """
        return prompt.format(conversation=conversation)
    
    def expand_scenario_prompt(self, scenario):
        prompt = """
        Based on the provided daily scenario, creatively generate a new
        and entirely different scenario. The new scenario must meet
        the following requirements:
        1. You may alter the action or subject of the original scenario.
        2. The new scenario should differ substantially from the
        original.
        3. Ensure the new scenario is realistic and feasible within a
        daily life context.
        4. Retain the same format as the original scenario.
        5. Limit your response to 10 words and present the new scenario
        in a single sentence.
        Original Scenario:
        {scenario}
        Modified Scenario:
        """
        return prompt.format(scenario=scenario)
    
    def atomic_task_generate_prompt(self, scenario):
        prompt = """
        You are training a model that can take a user's task description
        or query, and available functions as input, and generate a
        sequence of function calls to accomplish the task. Currently,
        you are generating basic atom tasks. Given a general life
        scenario as the context, please generate a basic atom task
        that can be accomplished in one step.
        Requirements of the task:
        1. The task should be a reasonable real life task based on the
        given scenario, and can be accomplished in one step.
        2. If you mention some information, criteria or constraints in
        the task, please give the details of these information,
        criteria or constraints. Do not assume the model has access
        to your personal information or prior knowledge, and it does
        not have chance to ask you for clarification.
        3. Please give enough details and make the task description as
        specific as possible, so the model can make deterministic
        function calls with deterministic arguments. Do not include
        any ambiguous or vague information.
        4. Do not mention specific tools or functions in the task
        description, and do not propose solutions, hints, or project
        outcomes.
        5. Limit the task description to 30 words, and avoid using
        adjectives and ambiguous words.
        Given Scenario:
        {scenario}
        Please give your response in one line directly, without any
        extra notation or format:
        """
        return prompt.format(scenario=scenario)
    
    def sequential_task_generate_prompt(self, task):
        prompt = """
        You are training a model that can take a user's task description
        or query, and available functions as input, and generate a
        sequence of function calls to accomplish the task. Currently,
        you are generating complex tasks for model training. Given a
        task, you need to add a subsequent task for this given task
        to make a more complex task.
        The requirements for the subsequent task are as follows:
        1. The subsequent task should use the output of the given task
        as input.
        2. The subsequent can only be conducted after the given task has
        been completed.
        3. The subsequent task and the given task can form a new
        composition task, and composing them can make a more
        complex multi-step task.
        ## Examples:
        ### Given Task: Give me a list of all the pets.
        ### Subsequent Task: What is the most common kind of pet in the
        list?
        ### Composition Task: Check the most common kind of pet in the
        list of all the pets.
        ### Given Task: Who is author of the book "The Great Gatsby"?
        ### Subsequent Task: When was the author of this book born?
        ### Composition Task: When was the author of the book "The Great
        Gatsby" born.
        ### Given Task: Give me the flight schedule from London to
        Edinburgh today.
        ### Subsequent Task: Which fight has the shortest duration?
        ### Composition Task: Give me the flight from London to
        Edinburgh with the shortest duration according to the flight
        schedule today.
        ### Given Task: Retrieve the headlines of the news today from
        BBC.
        ### Subsequent Task: What is the sentiment of the news
        respectively?
        ### Composition Task: What is the sentiment of each headline in
        today's news from BBC?
        ### Given Task: Which team won the World Cup in 2018?
        ### Subsequent Task: What is the team's captain?
        ### Composition Task: Who is the captain of the team that won
        the World Cup in 2018.
        ## Here is the given task, please give your response following
        the above format:
        ### Given Task: {task}
        """
        return prompt.format(task=task)
    
    def parallel_then_sequential_task_generate_prompt(self, task):
        prompt = """
        You are training a model that can take a user's task description
        or query, and available functions as input, and generate a
        sequence of function calls to accomplish the task. Currently,
        you are generating complex tasks for model training. Given a
        task, you need to add a paralle task and a subsequent task
        for this given task to make a more complex task.
        The requirements for the parallel task are as follows:
        1. The parallel task should be related to the given task, and
        the input should independent of the output of the given task.
        2. The parallel task can conduct at the same time as the given
        task, and they can be independent of each other.
        3. The output of the given task and the parallel task can be
        used together to conduct a subsequent task.
        The requirements for the subsequent task are as follows:
        1. The subsequent task should use the output of the given task
        and generate parallel task as input.
        2. The subsequent can only be conducted after the given task and
        the parallel task have been completed.
        3. The subsequent task, the given task and the parallel task can
        form a new composition task, and composing them can make a
        more complex multi-step task.
        ## Examples:
        ### Given Task: Give me a list of all the pets.
        ### Parallel Task: Find available pet food currently in the
        store.
        ### Subsequent Task: Check if the pet food is suitable for the
        pets in the list.
        ### Composition Task: Check if the pet food is suitable for the
        pets in the list of all the pets.
        ### Given Task: When was the author of the book "The Great
        Gatsby" born.
        ### Parallel Task: Find the publication date of the book "The
        Great Gatsby".
        ### Subsequent Task: When the book was published, how long had
        it been since the author was born?
        ### Composition Task: How old was the author of the book "The
        Great Gatsby" when the book was published?
        ### Given Task: Give me the flight schedule from London to
        Edinburgh today.
        ### Parallel Task: Find the every hour weather forecast in
        Edinburgh today.
        ### Subsequent Task: What is the weather condition when the
        first flight arrives?
        ### Composition Task: I am in London, and I want to know the
        weather condition when the first flight arrives in Edinburgh
        today.
        ### Given Task: What is the sentiment of each headline in today'
        s news from BBC?
        ### Parallel Task: Find the sentiment of each headline in today'
        s news from CNN.
        ### Subsequent Task: Which news source has more positive news
        today?
        ### Composition Task: Compare the sentiment of each headline in
        today's news from BBC and CNN, and check which news source
        has more positive news.
        ### Given Task: Who is the captain of the team that won the
        World Cup in 2018?
        ### Parallel Task: Who is the coach of the team that won the
        World Cup in 2018?
        ### Subsequent Task: Are the captain and the coach from the same
        country?
        ### Composition Task: Check if the captain and the coach of the
        team that won the World Cup in 2018 are from the same country
        .
        ## Here is the given task, please give your response following
        the above format:
        ### Given Task: {task}
        """
        return prompt.format(task=task)
    
    def filter_composition_task_prompt(self, task, sub_tasks):
        prompt = """
        You are an expert in task decomposition. Currently, you are
        given a composition task and its potential task breakdown.
        Please check if the sub-tasks can be used to complete the
        composition task.
        Composition task:
        {task}
        Potential task breakdown:
        {sub_tasks}
        Please check if the sub-tasks can be used to complete the
        composition task. You should first give your analysis and
        thinking, and finally give your conclusion (yes or no)
        enclosed in <ans>, for example, <ans>yes</ans> or <ans>no</ans>:
        """
        return prompt.format(task=task, sub_tasks=sub_tasks)
    
    def function_generate_prompt(self, task, sub_tasks):
        prompt = """
        You are training a model that can take a user's task description
        or query, and available functions as input, and generate a
        sequence of function calls to accomplish the task. Currently,
        you are generating the training data for this model.
        Given a composition task and its task breakdown, please
        generate corresponding aviliable functions that can be used
        to accomplish each sub-task, and finally the composition
        task can be accomplished by calling these functions
        sequentially.
        ## Requirements for the functions:
        1. The functions must possess a succinct, comprehensible name
        and description.
        2. The functions should not tailored for a current task, are to
        be used for other future tasks as well, hence the design of
        APIs should be sufficiently generalized.
        3. Avoid the recurrence of the task or its components in the
        function description and name, offering a generic perspective
        that can be employed across different contexts.
        4. Make every function sufficiently granular and independent,
        avoiding the conflation of multiple tasks within a single
        function and avert creating monolithic APIs.
        5. Consistency in terms of parameters and returns from each
        function is critical. For instance, if two functions are
        called sequentially, the output of the first should either
        align with or constitute a part of the input for the second
        function, irrespective of varying parameter terminologies.
        ## Requirements for the number of functions:
        1. One sub-task may need zero, one or multiple functions to
        complete it.
        2. If a sub-task is about logic, comparision, set operation or
        calculation, which can be solved by large language models,
        then no function is needed for this sub-task, just leave the
        func_list of this sub-task empty.
        ## Composition task:
        {task}
        ## Task breakdown:
        {sub_tasks}
        ## Response format:
        '''json
        [
        {{
        "sub_task": "a sub task from the task breakdown",
        "func_list": [
        {{
        "name": "<function name>",
        "description": "<function usage description>",
        "parameters": {{
        "<param1>": {{
        "type": "<can be string, number, boolean,
        object, array, enum and anyOf>",
        "description": "<param1 description>",
        ... <more keys if needed>
        }},
        ... <more parameters if needed>
        }},
        "required": "<array of required parameters, maybe
        not all parameters above are required>"
        "responses": {{
        "<res1>" {{
        "type": "<value1 type>",
        "description": "<value1 description>"
        }},
        "<res2>": {{
        "type": "<value2 type>",
        "description": "<value2 description>"
        }}
        }}
        }},
        {{
        ... <more functions if needed>
        }}
        ]
        }}
        ... <more sub tasks and corresponding functions if needed>
        ]
        '''
        ## Please respond following the format above:
        """
        return prompt.format(task=task, sub_tasks=sub_tasks)

    def user_agent_prompt(self, task):
        prompt = """
        Assume you are playing the role of a user engaging with an AI assistant in a multi-turn task-solving scenario.
        Currently, your goal is to complete a predefined task, and you
        are seeking the AI assistant for this purpose.
        **Task**
        {task}
        During this conversation, you should take on an active role and
        explore the AI assistant's capability to solve problems \
        within the **Task** using a series of function (tool) calls. You
        should adhere to the following guidelines:
        1. Your task involves a complex task requiring multiple steps to
        complete. In your initial question to the AI assistant, you
        should provide a detailed explanation of the task, including
        necessary information (such as potential data) that might be
        needed to solve the problem. However, you should withhold
        specific solution steps (e.g., avoid sequential terms like "
        firstly," "secondly") and not dictate which functions (tools)
        the AI should use - that is for the AI to determine.
        2. Remember, during this multi-turn dialogue, you are portraying
        the role of a human user. Your questions and responses
        should reflect this human aspect. All your outputs should
        enclose within "<human>" tag, for example, "<human> ... </
        human>".
        """
        return prompt.format(task=task)
    
    def assistant_agent_prompt(self, sub_task, sub_task_func):
        prompt = """
        You are simulating the role of an expert in using functions (i.e
        ., tools) to solve users' tasks. You already possess
        knowledge on how to decompose the task into subtasks and
        understand which tools to use for their resolution.
        **Subtasks**
        {sub_task}
        **Available Functions for Subtasks**
        {sub_task_func}
        Please use the tools provided above to answer the question posed
        by "<human>". You must try as much as possible to use these
        tools, instead of directly answering the question using your
        prior knowledge.
        Your response must obey the following format:
        Observation: Carefully observe the user "<human>"'s question as
        well as the output of the function call (often enclosed
        within the "<func_return>" tag). Be sure to check for any
        errors in previous outputs, as they may not always be
        accurate. Enclose your observation within the "<observation>"
        tag.
        Thought: After observing and combining the previously listed
        steps, give detailed and clear thoughts, reasonings, or
        reflections, and according to the plan decide the next step.
        Function Call: Name and arguments of the function call. The
        function name must be same as its name in above function list
        , and the arguments must obey the format required by the
        function. Enclose the function call within the "<func_call>"
        tag. If possible, you can call multiple functions in parallel
        , be sure the functions called  parallelly are independent of
        each other.
        Final Answer: When you believe the task is complete, you may
        use 'final_answer' to provide a detailed summary of the
        results to give to the user, enclose the final answer within
        the tag "<final>".
        Example 1 (regular function call):
        <observation> User has provided two numbers - 15 and 25. </
        observation>
        <thought> Based on user's request, we need to find the greatest
        common divisor of these two numbers. We can use the function
        'find_greatest_common_divisor' to solve this problem. </
        thought>
        <func_call>[
        {{
        "name": "find_greatest_common_divisor",
        "arguments": {{"num1": 15, "num2": 25}}
        }}
        ]</func_call>
        Example 2 (parallel function call):
        <observation> User wants to know the weather in two cities - New
        York and London. </observation>
        <thought> We can use the function 'get_weather' to find the
        weather in New York and London. And the call to this function
        can be done in parallel. </thought>
        <func_call>[
        {{
        "name": "get_weather",
        "arguments": {{"city": "New York"}}
        }},
        {{
        "name": "get_weather",
        "arguments": {{"city": "London"}}
        }}
        ]</func_call>
        Furthermore, when the user "<human>" raises a question, you need
        to provide a structured plan to solve the question ('
        structured' means that the plan needs to include steps in
        sequential order, such as Step 1, 2, 3, etc., or logic
        processes that include loops and decision branches). The
        contents of the plan can be placed in the first round
        response's <thought>, and try as much as possible to follow
        this plan in every subsequent function call. However, as
        necessary, you may also modify the relevant plans according
        to the result of the function call.
        """
        return prompt.format(sub_task=sub_task, sub_task_func=sub_task_func)
    
    def tool_agent_prompt(self, function):
        prompt = """
        You are simulating a computer system with powerful computational
        capabilities and a complete setup. You possess ample
        external prior knowledge, allowing you to run any arbitrary
        function and execute calls to produce results, and you never
        make errors. Give a following function, you should simulate
        the operation of a computer system program as closely as
        possible.
        **Function**
        {function}
        Given a function call, you should execute the function and
        provide the results in JSON format. Your response should
        directly provide the results in JSON format, should not
        contain irrelevant information, and must enclose within "<
        func_return>" tag.
        ### Example of function return:
        <func_call>
        {{
        "name": "get_weather",
        "arguments": {{"city": "New York"}}
        }}
        <func_return>
        {{
        "temperature": "25C",
        }}
        </func_return>
        """
        return prompt.format(function=function)
            