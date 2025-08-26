'''
A collection of prompts for the text2sql operator.
'''
import random
import numpy as np
import json
from typing import List


class SQLConsistencyFilterPrompt:
    def __init__(self):
        pass

    def build_prompt(self, question: str, sql: str, schema: str) -> str:
        prompt = f"""
        **Task Overview**
        Determine if the SQL query correctly answers the given question based on the provided schema.

        **Question**
        {question}

        **SQL**
        {sql}

        **Schema**
        {schema}

        **Evaluation Criteria**
        1. **Logical Alignment**: Does the SQL query logically address what the question is asking?
        2. **Schema Compliance**: Are the tables, columns, and relationships used correctly according to the schema?
        3. **Completeness**: Does the SQL capture all necessary conditions and requirements from the question?
        4. **Correctness**: Are there any logical errors that would prevent getting the correct answer?

        **Output Format**:
        The conclusion should be enclosed in a code block:
        ```
        <Conclusion> YES/NO </Conclusion>
        ```

        **Decision Rules**
        - YES: SQL correctly implements the question requirements
        - NO: SQL has logical errors or doesn't address the question properly
        - When uncertain about edge cases, explain the uncertainty in analysis but still provide a definitive YES/NO

        **Answer**
        Let's proceed step by step.
        """
        return prompt

class Text2SQLCotGeneratorPrompt:
    def __init__(self):
        pass

    def build_prompt(self, schema: str, question: str, sql: str) -> str:
        prompt = f"""
        You are a senior data analyst specializing in SQL. Your task is to translate a natural language question into an executable SQLite query, providing a detailed reasoning trace.

        You will also receive a reference solution from a colleague, which may or may not be correct. This extra information intends to help you generate your answer, but you are asked not to mention the reference solution in any form.
        The reference solution might include: 
        1. Unnecessary table and column selections. 
        2. Incorrect or excessive joins. 
        3. Misalignment with the question.
        4. Opportunities for simplification.

        Ensure the SQL query is presented in a Markdown code block with proper syntax highlighting, like this:
        ```sql
        SELECT * FROM table;
        ```

        [Database Schema]:
        {schema}

        [Natural Language Question]:
        {question}

        [Reference Solution]:
        ```sql
        {sql}
        ```

        Provide your step-by-step text-to-SQL solution here.
        """
        return prompt


class SelectSQLGeneratorPrompt:
    def __init__(self):
        self.simple_criterion = '''**Criteria:**
        Simple SQL queries may satisfy one or more of the following criteria:
        - Simple queries should select data from a single table only.
        - Basic aggregate functions are permitted, such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
        - No joins are allowed; the query must operate on a single table.

        **Example of Simple SQL Query:**
        ```sql
        SELECT name, department_name
        FROM employees
        WHERE level > 5
        ORDER BY age DESC;
        ```'''
    
        self.moderate_criterion = '''**Criteria:**
        Moderate SQL queries may satisfy one or more of the following criteria:
        - Involves table joins, such as `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, etc.
        - Includes subqueries within the `SELECT` or `WHERE` clauses.
        - Utilizes aggregate functions alongside a `GROUP BY` clause.
        - Contains complex `WHERE` conditions, including `IN`, `BETWEEN`, `LIKE`.
        - Incorporate a `HAVING` clause to filter aggregated results.
        - Uses aggregate functions like `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, etc.

        **Example of Moderate SQL Query:**
        ```sql
        SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
        FROM employees e
        INNER JOIN departments d ON e.department_id = d.department_id
        LEFT JOIN salaries s ON e.employee_id = s.employee_id
        WHERE e.age > 30 AND e.status = 'active'
        GROUP BY e.name, d.department_name
        HAVING AVG(s.salary) > 50000;
        ```'''

        self.complex_criterion = '''**Criteria:**
        Complex SQL queries may satisfy one or more of the following criteria:
        - Contains complex nested subqueries.
        - Utilizes multiple types of joins, including self-joins.
        - Includes window functions, such as `ROW_NUMBER`, `RANK`, etc.
        - Uses Common Table Expressions (CTEs) for improved readability.
        - Combines multiple aggregate functions.
        - Involves complex `WHERE` and `HAVING` clauses with multiple conditions.
        - Utilizes advanced functions and operators.

        **Example of Complex SQL Query:**
        ```sql
        WITH EmployeeCTE AS (
            SELECT employee_id, name, department_id, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
            FROM employees
        )
        SELECT e.name, d.department_name
        FROM EmployeeCTE e
        INNER JOIN departments d ON e.department_id = d.department_id
        WHERE e.rank <= 3;
        ```'''

        self.highly_complex_criterion = '''**Criteria:**
        Highly complex SQL queries may satisfy one or more of the following criteria:
        - Includes multiple Common Table Expressions (CTEs) for readability.
        - Combines nested subqueries and various joins.
        - Utilizes recursive CTEs for hierarchical or recursive queries.
        - Extensively uses advanced window functions.
        - May involve `UNION` or `UNION ALL` to combine result sets.
        - Implements complex logic with advanced analytical functions.
        - Employs a wide range of SQL clauses and conditions.
        - Utilizes a broad spectrum of SQL functions and advanced features.

        **Example of Highly Complex SQL Query:**
        ```sql
        WITH RECURSIVE EmployeeHierarchy AS (
            SELECT employee_id, name, manager_id, department_id, 1 as level
            FROM employees
            WHERE manager_id IS NULL
            UNION ALL
            SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
            FROM employees e
            JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
        ),
        DepartmentSalaries AS (
            SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
            FROM EmployeeHierarchy eh
            INNER JOIN departments d ON eh.department_id = d.department_id
            INNER JOIN salaries s ON eh.employee_id = s.employee_id
        ),
        DepartmentStats AS (
            SELECT 
                d.department_id,
                COUNT(e.employee_id) AS employee_count,
                AVG(s.salary) AS average_salary
            FROM employees e
            INNER JOIN salaries s ON e.employee_id = s.employee_id
            INNER JOIN departments d ON e.department_id = d.department_id
            GROUP BY d.department_id
        )
        SELECT ds.name, ds.level, 
            SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
        FROM DepartmentSalaries ds
        INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
        ORDER BY ds.level, ds.name;
        ```'''
        self.complexity2criterion = {
            "Simple": self.simple_criterion,
            "Moderate": self.moderate_criterion,
            "Complex": self.complex_criterion, 
            "Highly Complex": self.highly_complex_criterion
        }
        
        random.seed(42)

    def sql_func_template(self, sql_funcs: str) -> str:
        template = """### SQL Functions
        You may consider one or more of the following SQL functions while generating the query:
        {sql_funcs}
        Important tips:
        Except for the functions listed above, you may use any other functions as long as they conform to the syntax of the database engine.
        """
        return template.format(sql_funcs=sql_funcs)

    def insert_stmts_template(self, insert_statements: str) -> str:
        template = '''### INSERT INTO Statements
        Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:
        {insert_statements}
        '''
        return template.format(insert_statements=insert_statements)

    def sql_synthesis_prompt(self, schema_str: str, sql_function_prompt: str, db_value_prompt: str, complexity: str, criterion: str, db_engine: str, column_count: int) -> str:
        template = '''**Task Overview**
        Create an executable SQL query based on the provided information.

        **Database Schema**
        {schema_str}

        {sql_function_prompt}

        {db_value_prompt}

        **SQL Query Complexity**
        Ensure the SQL query matches the {complexity} level, defined as follows:
        {criterion}

        **Output Format Requirements**
        Enclose the SQL query in a code block:
        ```sql
        -- Your SQL query here
        ```

        **SQL Query Requirements**
        1. Use the syntax specific to the {db_engine} database engine.
        2. Incorporate advanced functions if appropriate, but they are not mandatory.
        3. Address real-world data analysis needs. Avoid trivial or nonsensical queries.
        4. (Very important) Ensure the final SQL query selects {column_count} columns.

        **Answer**
        Let's proceed step by step.
        '''
        return template.format(
            schema_str=schema_str,
            sql_function_prompt=sql_function_prompt.strip(),
            db_value_prompt=db_value_prompt.strip(),
            complexity=complexity,
            criterion=criterion.strip(),
            db_engine=db_engine,
            column_count=column_count
        )

    def build_prompt(self, insert_statements: List[str], functions: List[str], db_engine: str, create_statements: List[str]) -> tuple[str, str]:
        complexity = random.choice(["Simple", "Moderate", "Complex", "Highly Complex"])

        if len(insert_statements) == 0:
            db_value_prompt = ""
        else:
            if len(insert_statements) > 4:
                insert_statements = random.sample(insert_statements, 4)
            db_value_prompt = self.insert_stmts_template(
                insert_statements="\n\n".join(insert_statements)
            )

        function_num = random.randint(0, 2)
        if function_num == 0:
            sql_function_prompt = "### SQL Functions\nYou can use any function supported by the database engine."
        else:
            sql_funcs = ""
            sampled_functions = random.sample(functions, min(function_num, len(functions)))
            for idx, func in enumerate(sampled_functions):
                sql_funcs += f"Function {idx + 1}:\n{func.strip()}\n"
            sql_function_prompt = self.sql_func_template(sql_funcs=sql_funcs)

        column_count = np.random.geometric(0.6, 1)[0]
        prompt = self.sql_synthesis_prompt(
            schema_str="\n\n".join(create_statements),
            sql_function_prompt=sql_function_prompt.strip(),
            db_value_prompt=db_value_prompt.strip(),
            complexity=complexity,
            criterion=self.complexity2criterion[complexity].strip(),
            db_engine=db_engine,
            column_count=column_count
        )
        return prompt, complexity

class Text2SQLQuestionGeneratorPrompt:
    def __init__(self):
        pass

    def get_style2desc(self):
        template = {
        "Formal": '''**Formal Style**
        - Uses standard grammar and vocabulary.
        - Example: Find all students older than 18 years and return their home addresses.''',

        "Colloquial": '''**Colloquial Style**
        - Employs informal vocabulary and expressions.
        - Example: Hey! Could you help me find all the students who are over 18? I'd love to know their names and where they live.''',

        "Imperative": '''**Imperative Style**
        - Uses command or directive sentences.
        - Example: Could you please gather all the students who are older than 18? I really need to know their names and where they live!''',

        "Interrogative": '''**Interrogative Style**
        - Uses question forms.
        - Example: Could you tell me which students are older than 18 and what their home addresses are?''',

        "Descriptive": '''**Descriptive Style**
        - Uses detailed descriptions with contextual information.
        - Example: I want to know the names and home addresses of all students older than 18.''',

        "Concise": '''**Concise Style**
        - Use short sentences.
        - Example: Students older than 18, return their names and addresses.''',

        "Vague": '''**Vague Style**
        - Includes ambiguous vocabulary requiring inference.
        - Example: What are the names and addresses of those older students? (External Knowledge: 'older students' refers to age >= 18.)''',

        "Metaphorical": '''**Metaphorical Style**
        - Uses metaphors or metaphorical expressions.
        - Example: Find the names and addresses of those who have reached adulthood. (External Knowledge: 'reached adulthood' refers to age >= 18.)'''
        }
        return template

    def get_steps_wo_ek(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.'''
        return template

    def get_steps_w_ek(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Question:** Formulate a natural language question based on the SQL query and explanation.
        3. **External Knowledge:** For Vague or Metaphorical styles, include external knowledge to enhance clarity.'''
        return template

    def get_steps_multi_round(self):
        template = '''1. **Explain the SQL Query:** Provide a detailed explanation of what the query does.
        2. **Generate a Dialogue:** Create a conversation between the User and the Assistant based on the SQL query and its explanation.'''
        return template

    def get_guidelines_wo_ek(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''
        return template

    def get_guidelines_w_ek(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the natural language question accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.
        3. If necessary, incorporate external knowledge using multiple entries separated by semicolons (";"). These can include formulas, common sense, domain-specific knowledge, or extended context, such as information from long documents. Each entry should be concise.'''
        return template

    def get_guidelines_multi_round(self):
        template = '''1. Clearly describe the columns being selected by the SQL query. For example:
        - "SELECT * ... FROM ..." means "Find all ...";
        - "SELECT f.check_date, f.status, f.remarks, c.year, c.year_min, c.year_max, c.year_average, c.data_quality_score FROM ..." means "Return the check dates, statuses, remarks, years, minimum years, maximum years, average years, and quality scores for ...".
        2. Ensure the conversation accurately captures the semantics of the SQL query, including conditions such as predicates, `ORDER BY`, and `LIMIT` clauses.'''
        return template

    def get_output_format_wo_ek(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question)
        [QUESTION-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].'''
        return template

    def get_output_format_w_ek(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question)
        [QUESTION-END]

        [EXTERNAL-KNOWLEDGE-START]
        (External Knowledge)
        [EXTERNAL-KNOWLEDGE-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Translate the SQL query into a natural language question, enclosed within [QUESTION-START] and [QUESTION-END].
        - **External Knowledge**: Include any relevant external knowledge if applicable, enclosed within [EXTERNAL-KNOWLEDGE-START] and [EXTERNAL-KNOWLEDGE-END]. Leave this section blank if not needed.'''
        return template

    def get_output_format_multi_round(self):
        template = '''Please structure your response as follows:

        [EXPLANATION-START]
        (SQL Explanation)
        [EXPLANATION-END]

        [QUESTION-START]
        (Natural Language Question, in the format of [{"User": ...}, {"Assistant": ...}, {"User": ...}, ....])
        [QUESTION-END]

        - **SQL Explanation**: Provide a clear and detailed explanation of the SQL query, enclosed within [EXPLANATION-START] and [EXPLANATION-END].
        - **Natural Language Question**: Convert the SQL query into a multi-round dialogue, enclosed within [QUESTION-START] and [QUESTION-END]. Represent this as a list that captures multiple rounds of conversation between the User and the Assistant.'''
        return template

    def get_instruction_wo_ek(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation and the question corresponding to the SQL query."
        return template

    def get_instruction_w_ek(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation, the question, and the external knowledge corresponding to the SQL query."
        return template

    def get_instruction_multi_round(self):
        template = "Based on the above information, follow the reasoning steps to generate the explanation and the dialogue corresponding to the SQL query."
        return template

    def question_synthesis_prompt(self, style_desc, engine, column_info, sql, steps, guidelines, output_format, instruction):


        template = '''**Task Overview**
        Your task is to create a high-quality natural language question based on a given SQL query and other information.

        **Style**
        The natural language question should follow this style:
        {style_desc}

        **Database Engine**
        {engine}

        **Column Information**
        Below are column names and their corresponding descriptions:
        {column_info}

        **SQL Query**
        Given SQL query:
        ```sql
        {sql}
        ```

        **Reasoning Steps**
        {steps}

        **Guidelines**
        {guidelines}

        **Output Format**
        {output_format}

        **Insturction**
        {instruction}
        '''
        return template.format(
            style_desc = style_desc,
            engine = engine,
            column_info = column_info,
            sql = sql,
            steps = steps,
            guidelines = guidelines,
            output_format = output_format,
            instruction = instruction
        )  

    def build_prompt(self, data, input_db_id_key, input_sql_key, styles, db_id2column_info, db_type) -> str:
        style_name = random.sample(styles, 1)[0]
        column_name2column_desc = db_id2column_info[data[input_db_id_key]]
        used_column_name2column_desc = dict()
            
        for column_name, column_desc in column_name2column_desc.items():
            if column_name.lower() in data[input_sql_key].lower():
                used_column_name2column_desc[column_name] = column_desc

        if style_name in ["Vague", "Metaphorical"]:
            steps = self.get_steps_w_ek()
            guidelines = self.get_guidelines_w_ek()
            instruction = self.get_instruction_w_ek()
            output_format = self.get_output_format_w_ek()
        else:
            steps = self.get_steps_wo_ek()
            guidelines = self.get_guidelines_wo_ek()
            instruction = self.get_instruction_wo_ek()
            output_format = self.get_output_format_wo_ek()

        prompt = self.question_synthesis_prompt(
            style_desc=self.get_style2desc()[style_name].strip(),
            engine=db_type,
            column_info=json.dumps(used_column_name2column_desc, indent=2, ensure_ascii=False).strip(),
            sql=data[input_sql_key].strip(),
            steps=steps.strip(),
            guidelines=guidelines.strip(),
            output_format=output_format.strip(),
            instruction=instruction.strip()
        )

        return prompt, style_name


class SQLVariationGeneratorPrompt:
    def __init__(self):
        pass

    def variation_type_prompt(self, variation_type: int):
        type_prompts = [
            '''
            Data Value Transformations
            - Alter filter conditions, date ranges, or numerical thresholds
            - Change sorting criteria or limit values
            - Modify aggregation boundaries (e.g., GROUP BY different time periods)
            ''',

            '''Query Structure Modifications
            - Convert aggregation queries to window functions or vice versa
            - Change from simple queries to subqueries or CTEs
            - Transform JOINs to EXISTS/IN clauses or vice versa
            - Switch between correlated and non-correlated subqueries
            ''',

            '''Business Logic Changes
            - Adapt the query for different business scenarios (sales → inventory, customers → suppliers)
            - Modify to handle different data granularities (daily → monthly, individual → grouped)
            - Change the analytical perspective (profit analysis → cost analysis)
            - Alter the metrics being calculated (sum → average, count → percentage)
            ''',

            '''Complexity Enhancements
            - Add extra filtering conditions or business rules
            - Introduce additional table joins
            - Include conditional logic with CASE statements
            - Add data validation or quality checks
            ''',

            '''Advanced SQL Features
            - Implement complex window functions with partitioning
            - Create queries requiring UNION/INTERSECT/EXCEPT operations
            - Add recursive CTEs for hierarchical data
            - Include pivot/unpivot operations
            ''',

            '''Performance and Optimization
            - Add performance optimization hints
            - Restructure for better index usage
            - Convert to more efficient query patterns
            - Add appropriate WHERE clause optimizations
            ''',
        ]
        return type_prompts[variation_type]

    def insert_stmts_template(self, insert_statements):
        template = '''### INSERT INTO Statements
        Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:
        {insert_statements}
        '''
        return template.format(insert_statements=insert_statements)

    def sql_variation_prompt(self, original_sql, schema_str, db_value_prompt, variation_prompt, db_engine):
        template = """**Task Overview**
        Create a new reasonable and executable SQL query by applying the specified transformations to the original query.

        **Database Engine**
        {db_engine}

        **Database Schema**
        {schema_str}

        {db_value_prompt}

        **Original SQL Query**
        ```sql
        {original_sql}
        ```

        **Transformation Instructions**
        {variation_prompt}

        **Requirements**
        1. The new query must be syntactically correct for {db_engine}
        2. All referenced tables/columns must exist in the provided schema
        3. Ensure the query is executable

        **Output Format**
        The transformed SQL query should be enclosed in a code block:
        ```sql
        -- Your transformed SQL query here
        ```

        **Answer**
        Let's proceed step by step.
        """
        return template.format(
            variation_prompt=variation_prompt,
            schema_str=schema_str,
            db_value_prompt=db_value_prompt,
            original_sql=original_sql,
            db_engine=db_engine
        )

    def build_prompt(self, original_sql, schema_str, db_value_prompt, db_engine) -> str:
        if len(db_value_prompt) == 0:
            db_value_prompt = ""
        else:
            if len(db_value_prompt) > 4:
                db_value_prompt = random.sample(db_value_prompt, 4)
            db_value_prompt = self.insert_stmts_template(
                insert_statements="\n\n".join(db_value_prompt)
            )

        variation_type = random.randint(0, 5)
        variation_prompt = self.variation_type_prompt(variation_type=variation_type)
                    
        prompt = self.sql_variation_prompt(
            original_sql=original_sql,
            schema_str=schema_str,
            db_value_prompt=db_value_prompt.strip(),
            variation_prompt=variation_prompt.strip(),
            db_engine=db_engine
        )
        return prompt

class Text2SQLPromptGeneratorPrompt:
    def __init__(self):
        pass

    def build_prompt(self, schema: str, question: str) -> str:
        prompt = '''Task Overview:
            /* Given the following database schema: */
            {schema}
            /* Answer the following: {question} */
            Let's think step by step'''
        return prompt.format(schema=schema, question=question)