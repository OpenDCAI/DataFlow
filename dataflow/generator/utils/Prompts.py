'''
A collection of prompts for the text generator.
Every algorithm should contain its prompts in a class in this file.
'''

class CodeScorerPrompt:
    '''
    The prompt for the code scorer.
    '''
    def __init__(self):
        pass

    def code_scorer_prompt(self) -> str:
        system_prompt = '''You are an advanced grading assistant responsible for evaluating user responses. Given a question, a solution, and an analysis, you must analyze and provide structured feedback in two sections, **Grading** and **Feedback**.

1. **Grading**
Assign a score between 1 and 5, where:
5 = Perfect answer, fully correct and optimal
4 = Mostly correct, minor issues that do not significantly affect correctness
3 = Partially correct, but has notable errors or inefficiencies
2 = Mostly incorrect, but contains some relevant elements
1 = Incorrect or fundamentally flawed
2. **Feedback**
Provide a brief, actionable hint to guide the user toward improvement without giving away the solution.
If applicable, suggest best practices or alternative approaches to enhance understanding.
Below is a format reference
**Grading**: 4

**Feedback**: Your solution is mostly correct and demonstrates a solid understanding of how to handle validation errors within an ASP.NET MVC application. However, ensure that the method clearly distinguishes between handling validation exceptions and general exceptions. Additionally, consider implementing logging within both exception handling scenarios to improve error traceability. To enhance your approach, familiarize yourself with utilizing built-in logging mechanisms in ASP.NET for improved maintainability. 
'''
        return system_prompt
    
class CodeRefinerPrompt:
    '''
    The prompt for the code refiner.
    '''
    def __init__(self):
        pass

    def code_refiner_prompt(self, code) -> str:
        prompt = '''You are an excellent programmer. Please improve the following code according to these instructions:


1. Fix all indentation issues.
2. Rename unclear or non-descriptive variable names (e.g., `a`, `b`, `x`) to meaningful and descriptive names.
3. Keep the original logic and structure unchanged. In other words, you should not change or add anything except the indentation and variable names.
4. Return only the fully optimized code. Do not include any explanation, comments, or extra text.

Here is the code you need to refine:
'''
        return prompt + code + f'Remeber to return the refined code only.'
    
'''
A collection of prompts for the text generator.
Every algorithm should contain its prompts in a class in this file.
'''

class CodeScorerPrompt:
    '''
    The prompt for the code scorer.
    '''
    def __init__(self):
        pass

    def code_scorer_prompt(self) -> str:
        system_prompt = '''You are an advanced grading assistant responsible for evaluating user responses. Given a question, a solution, and an analysis, you must analyze and provide structured feedback in two sections, **Grading** and **Feedback**.

1. **Grading**
Assign a score between 1 and 5, where:
5 = Perfect answer, fully correct and optimal
4 = Mostly correct, minor issues that do not significantly affect correctness
3 = Partially correct, but has notable errors or inefficiencies
2 = Mostly incorrect, but contains some relevant elements
1 = Incorrect or fundamentally flawed
2. **Feedback**
Provide a brief, actionable hint to guide the user toward improvement without giving away the solution.
If applicable, suggest best practices or alternative approaches to enhance understanding.
Below is a format reference
**Grading**: 4

**Feedback**: Your solution is mostly correct and demonstrates a solid understanding of how to handle validation errors within an ASP.NET MVC application. However, ensure that the method clearly distinguishes between handling validation exceptions and general exceptions. Additionally, consider implementing logging within both exception handling scenarios to improve error traceability. To enhance your approach, familiarize yourself with utilizing built-in logging mechanisms in ASP.NET for improved maintainability. 
'''
        return system_prompt
    


class OssInstGeneratorPrompt:
    '''
    The prompt for the oss inst generator.
    '''
    def __init__(self):
        pass
    
    def oss_inst_generator_prompt(self, code: str) -> str:
        return f'''
        You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions. Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in three distinct sections: **Problem Description**, **Analysis**, and **Solution**. 

        Guidelines for each section:  
        1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.  
        2. **Analysis**: Offer a clear, logical breakdown of the problem, detailing the steps and thought processes required to solve it. This section should help understand the reasoning behind the solution.  
        3. **Solution**: Provide a comprehensive, correct solution that addresses the **Problem Description**. Ensure the solution is in the form of a self-contained script that does not rely on any assumptions or external details not provided in the **Problem Description**.  
        **Code Snippet** 
        {code}
        '''

class AnswerGeneratorPrompt:
    '''
    The prompt for the answer generator.
    '''
    def __init__(self):
        pass

    def Classic_COT_Prompt(question: str) -> str:
        """
        为给定数学题目生成系统提示信息
        """
        prompt = (
            r'''You are an intelligent chatbot designed for writing the answer of the given math question.
    Remember: DO NOT output anything else, only output the answer you make.
    Generate a solution of a given math problem strictly following this format:
    1. Identify key components of the problem
    2. Apply theorems/formulas with step-by-step derivation
    3. Perform calculations with intermediate verification
    4. Final answer in \boxed{} notation

    Format Requirements:
    - Prefix each step with "→" (use the actual arrow symbol, not its Unicode escape sequence)
    - Ensure all symbols and special characters are presented using LaTeX commands where appropriate (e.g., ≥ as \\geq, ÷ as \\div)

    Example Template:
    Problem: Find the minimum value of function f(x) = x³ - 3x² + 4 on interval [-1, 3]

    Solution:
    1. Find critical points:
    → f'(x) = 3x² - 6x
    → Set derivative to zero: 3x(x-2) = 0 ⇒ x=0, x=2

    2. Evaluate function at critical points and endpoints:
    → f(-1) = (-1)^3 - 3(-1)^2 + 4 = -1 -3 +4 = 0.0000
    → f(0) = 0³ - 3(0)² +4 = 4.0000
    → f(2) = 8 - 12 +4 = 0.0000
    → f(3) = 27 - 27 +4 = 4.0000

    3. Compare values:
    → Minimum occurs at x=-1 and x=2

    Verification:
    → Second derivative test: f''(x) = 6x-6
    → f''(-1) = -12 < 0 (local max)
    → f''(2) = 6 > 0 (local min)

    \boxed{0}

    Here is the given problem you need to solve:
    '''
        )
        return prompt + question + r'''Your response must directly start with "Solution:" without any preamble, After the answer is generated finish your response right away.'''
    

class QuestionSynthesisPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self,items, question):
        prompt = f"""
        Create a new reasonable and solvable math problem from the original problem by applying some of the following transformations(focus on all the transformations of "{items}"):

        1. Alter numerical values or expressions, ensuring the new problem remains solvable.
        2. Modify the problem type: introduce concepts like ratios or percentages, switch between derivatives and integrals, change the question from finding an area to finding a perimeter, etc.
        3. Contextualize the problem within a real-world scenario, such as incorporating various payment methods or deferred payments with interest.
        4. Add additional premises that require considering an extra factor separately in solving the problem.
        5. Increase the complexity of the problem by introducing multiple conditions that necessitate case-by-case analysis for a solution.

        Here is the problem from the user:
        {question}
        Write another problem inspired by this one.
        Not only change the problem scenario, but also try to create a new problem that requires another approach to solve.
        Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
        After the problem is generated finish your response right away.
        """
        return prompt
    
class QuestionCategoryPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self, question):
        prompt = f"""
        You are a classification assistant specialized in mathematics. Your task is to classify the given text into one primary category and one secondary category according to the following taxonomy. Do not output any extra explanation. Return only a JSON object with the keys "primary_category" and "secondary_category".

        Taxonomy:
        1. Foundations and Logic
        - 1.1 Mathematical Logic and Set Theory
        - 1.2 Basic Theory, Formalization, and History & Education

        2. Algebra and Number Theory
        - 2.1 Linear Algebra and Group Theory
        - 2.2 Ring Theory, Field Theory, and Polynomial Algebra
        - 2.3 Commutative Algebra and Homological/Categorical Methods
        - 2.4 Number Theory
        - 2.5 Algebraic Geometry

        3. Analysis and Differential Equations
        - 3.1 Real Analysis, Measure Theory, and Functional Analysis
        - 3.2 Complex Analysis and Special Functions
        - 3.3 Differential Equations and Dynamical Systems
        - 3.4 Integral Transforms, Integral Equations, and Difference Equations
        - 3.5 Harmonic Analysis

        4. Geometry and Topology
        - 4.1 Euclidean, Analytic, and Convex/Discrete Geometry
        - 4.2 Differential Geometry and Manifold Theory
        - 4.3 Topology and Algebraic Topology

        5. Probability, Statistics, and Discrete Mathematics
        - 5.1 Probability Theory and Stochastic Processes
        - 5.2 Mathematical Statistics
        - 5.3 Combinatorics and Graph Theory

        6. Applied and Computational Mathematics
        - 6.1 Numerical Analysis and Computational Methods
        - 6.2 Optimal Control, Variational Methods, and Optimization
        - 6.3 Operations Research and Game Theory
        - 6.4 Systems Theory and Control
        - 6.5 Computer Science and Algorithms
        - 6.6 Mathematical Physics and Engineering Mathematics
        - 6.7 Information and Communication
        - 6.8 Biomathematics

        7. Arithmetic
        - 7.1 Basic Arithmetic and Number Operations
        - 7.2 Word Problems and Real-Life Applications

        Classify the following text into one primary category and one secondary category based on the taxonomy above. The text is:
        {question}
        """
        return prompt

    
class QuestionDifficultyPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self, question):
        prompt = r"""
        # CONTEXT #
        I am a teacher, and I have some high-level olympiad math problems. 
        I want to evaluate the difficulty of these math problems. There are some references available regarding the difficulty of the problems:
        <difficulty reference>
        For reference, here are some sample problems from each of the difficulty levels 1-10:

        1: Jamie counted the number of edges of a cube, Jimmy counted the numbers of corners, and Judy counted the number of faces. They then added the three numbers. What was the resulting sum? (2003 AMC 8, Problem 1)

        1: How many integer values of $x$ satisfy $|x| < 3\pi$? (2021 Spring AMC 10B, Problem 1)

        1.5: A number is called flippy if its digits alternate between two distinct digits. For example, $2020$ and $37373$ are flippy, but $3883$ and $123123$ are not. How many five-digit flippy numbers are divisible by $15?$ (2020 AMC 8, Problem 19)

        2: A fair $6$-sided die is repeatedly rolled until an odd number appears. What is the probability that every even number appears at least once before the first occurrence of an odd number? (2021 Spring AMC 10B, Problem 18)

        2.5: $A$, $B$, $C$ are three piles of rocks. The mean weight of the rocks in $A$ is $40$ pounds, the mean weight of the rocks in $B$ is $50$ pounds, the mean weight of the rocks in the combined piles $A$ and $B$ is $43$ pounds, and the mean weight of the rocks in the combined piles $A$ and $C$ is $44$ pounds. What is the greatest possible integer value for the mean in pounds of the rocks in the combined piles $B$ and $C$? (2013 AMC 12A, Problem 16)

        3: Triangle $ABC$ with $AB=50$ and $AC=10$ has area $120$. Let $D$ be the midpoint of $\overline{AB}$, and let $E$ be the midpoint of $\overline{AC}$. The angle bisector of $\angle BAC$ intersects $\overline{DE}$ and $\overline{BC}$ at $F$ and $G$, respectively. What is the area of quadrilateral $FDBG$? (2018 AMC 10A, Problem 24)

        3.5: Find the number of integer values of $k$ in the closed interval $[-500,500]$ for which the equation $\log(kx)=2\log(x+2)$ has exactly one real solution. (2017 AIME II, Problem 7)

        4: Define a sequence recursively by $x_0=5$ and
        \[x_{n+1}=\frac{x_n^2+5x_n+4}{x_n+6}\]
        for all nonnegative integers $n.$ Let $m$ be the least positive integer such that
        \[x_m\leq 4+\frac{1}{2^{20}}.\]
        In which of the following intervals does $m$ lie?

        $\textbf{(A) } [9,26] \qquad\textbf{(B) } [27,80] \qquad\textbf{(C) } [81,242]\qquad\textbf{(D) } [243,728] \qquad\textbf{(E) } [729,\infty)$  
        (2019 AMC 10B, Problem 24 and 2019 AMC 12B, Problem 22)

        4.5: Find, with proof, all positive integers $n$ for which $2^n + 12^n + 2011^n$ is a perfect square. (USAJMO 2011/1)

        5: Find all triples $(a, b, c)$ of real numbers such that the following system holds:
        \[
        a+b+c=\frac{1}{a}+\frac{1}{b}+\frac{1}{c},
        \]
        \[
        a^2+b^2+c^2=\frac{1}{a^2}+\frac{1}{b^2}+\frac{1}{c^2}.
        \]
        (JBMO 2020/1)

        5.5: Triangle $ABC$ has $\angle BAC = 60^{\circ}$, $\angle CBA \leq 90^{\circ}$, $BC=1$, and $AC \geq AB$. Let $H$, $I$, and $O$ be the orthocenter, incenter, and circumcenter of $\triangle ABC$, respectively. Assume that the area of pentagon $BCOIH$ is the maximum possible. What is $\angle CBA$? (2011 AMC 12A, Problem 25)

        6: Let $\triangle ABC$ be an acute triangle with circumcircle $\omega,$ and let $H$ be the intersection of the altitudes of $\triangle ABC.$ Suppose the tangent to the circumcircle of $\triangle HBC$ at $H$ intersects $\omega$ at points $X$ and $Y$ with $HA=3,\ HX=2,$ and $HY=6.$ The area of $\triangle ABC$ can be written in the form $m\sqrt{n},$ where $m$ and $n$ are positive integers, and $n$ is not divisible by the square of any prime. Find $m+n.$ (2020 AIME I, Problem 15)

        6.5: Rectangles $BCC_1B_2,$ $CAA_1C_2,$ and $ABB_1A_2$ are erected outside an acute triangle $ABC.$ Suppose that
        \[\angle BC_1C+\angle CA_1A+\angle AB_1B=180^{\circ}.\]
        Prove that lines $B_1C_2,$ $C_1A_2,$ and $A_1B_2$ are concurrent. (USAMO 2021/1, USAJMO 2021/2)

        7: We say that a finite set $\mathcal{S}$ in the plane is balanced if, for any two different points $A$, $B$ in $\mathcal{S}$, there is a point $C$ in $\mathcal{S}$ such that $AC=BC$. We say that $\mathcal{S}$ is centre-free if for any three points $A$, $B$, $C$ in $\mathcal{S}$, there is no point $P$ in $\mathcal{S}$ such that $PA=PB=PC$.

        Show that for all integers $n\geq 3$, there exists a balanced set consisting of $n$ points.
        Determine all integers $n\geq 3$ for which there exists a balanced centre-free set consisting of $n$ points.
        (IMO 2015/1)

        7.5: Let $\mathbb{Z}$ be the set of integers. Find all functions $f : \mathbb{Z} \rightarrow \mathbb{Z}$ such that
        \[
        xf(2f(y)-x)+y^2f(2x-f(y))=\frac{f(x)^2}{x}+f(yf(y))
        \]
        for all $x, y \in \mathbb{Z}$ with $x \neq 0$. (USAMO 2014/2)

        8: For each positive integer $n$, the Bank of Cape Town issues coins of denomination $\frac1n$. Given a finite collection of such coins (of not necessarily different denominations) with total value at most $99+\frac{1}{2}$, prove that it is possible to split this collection into $100$ or fewer groups, such that each group has total value at most $1$. (IMO 2014/5)

        8.5: Let $I$ be the incentre of acute triangle $ABC$ with $AB\neq AC$. The incircle $\omega$ of $ABC$ is tangent to sides $BC, CA$, and $AB$ at $D, E,$ and $F$, respectively. The line through $D$ perpendicular to $EF$ meets $\omega$ at $R$. Line $AR$ meets $\omega$ again at $P$. The circumcircles of triangle $PCE$ and $PBF$ meet again at $Q$.

        Prove that lines $DI$ and $PQ$ meet on the line through $A$ perpendicular to $AI$. (IMO 2019/6)

        9: Let $k$ be a positive integer and let $S$ be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of $S$ around the circle such that the product of any two neighbors is of the form $x^2+x+k$ for some positive integer $x$. (IMO 2022/3)

        9.5: An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from $1$ to $10$.
        \[
        \begin{array}{ c@{\hspace{4pt}}c@{\hspace{4pt}} c@{\hspace{4pt}}c@{\hspace{2pt}}c@{\hspace{2pt}}c@{\hspace{4pt}}c }
        & & & 4 & & & \\
        & & 2 & & 6 & & \\
        & 5 & & 7 & & 1 & \\
        8 & & 3 & & 10 & & 9 \\
        \end{array}
        \]
        Does there exist an anti-Pascal triangle with $2018$ rows which contains every integer from $1$ to $1 + 2 + 3 + \dots + 2018$? (IMO 2018/3)

        10: Prove that there exists a positive constant $c$ such that the following statement is true: Consider an integer $n > 1$, and a set $\mathcal S$ of $n$ points in the plane such that the distance between any two different points in $\mathcal S$ is at least 1. It follows that there is a line $\ell$ separating $\mathcal S$ such that the distance from any point of $\mathcal S$ to $\ell$ is at least $cn^{-1/3}$.

        (A line $\ell$ separates a set of points S if some segment joining two points in $\mathcal S$ crosses $\ell$.) (IMO 2020/6)
        ## Some known difficulty ratings of the competitions.
        
        </difficulty reference>

        # OBJECTIVE #
        A. Summarize the math problem in a brief sentence, describing the concepts involved in the math problem.
        B. Based on the source of the given problem, as well as the difficulty of the problems referenced in these materials and the solution to the current problem, please provide 
        an overall difficulty score for the current problem. The score should be a number between 1 and 10, with increments of 0.5, and should align perfectly with the materials.
        # STYLE #
        Data report.
        # TONE #
        Professional, scientific.
        # AUDIENCE #
        Students. Enable them to better understand the difficulty of the math problems.
        # RESPONSE: MARKDOWN REPORT #
        ## Summarization
        [Summarize the math problem in a brief paragraph.]
        ## Difficulty
        [Rate the difficulty of the math problem and give the reason.]
        # ATTENTION #- Add "=== report over ===" at the end of the report.
        <example math problem>
        The problem requires finding the missing value in the equation

        \[
        \frac{1}{9}+\frac{1}{18}=\frac{1}{\square}.
        \]

        In other words, determine the number that should replace the square such that the sum of the fractions on the left equals the fraction on the right.
        </example math problem>
        ## Summarization
        The problem requires finding a value that makes the equation $\\frac{1}{9}+\\frac{1}{18}=\\frac{1}{\\square}$. 
        This involves adding two fractions and determining the equivalent fraction.
        ## Difficulty
        Rating: 1
        Reason: This problem is straightforward and primarily involves basic fraction addition, making it suitable for early middle school students. 
        === report over ===
        </example math problem>
        Let $\mathcal{P}$ be a convex polygon with $n$ sides, $n\ge3$. Any set of $n - 3$ diagonals of $\mathcal{P}$ that do not intersect in the interior of the polygon determine a triangulation of $\mathcal{P}$ into $n - 2$ triangles. If $\mathcal{P}$ is regular and there is a triangulation of $\mathcal{P}$ consisting of only isosceles triangles, find all the possible values of $n$. 
        </example math problem>
        ## Summarization
        The problem asks for the possible values of $n$ for a regular n-sided polygon that can be completely triangulated into isosceles triangles using non-intersecting diagonals. 
        The solution involves analyzing the properties of the diagonals forming isosceles triangles and deducing that $n$ can be expressed in terms of powers of 2.
        ## Difficulty
        Rating: 7
        Reason: The problem involves understanding properties of isosceles triangles in the context of polygon triangulation and requires critical reasoning to establish 
        relationships between the number of sides and powers of 2, making it more complex than typical undergraduate-level problems.
        === report over ===
        <math problem>
        [Question]: \n

        """

        return prompt + question


class CodeCommentGeneratorPrompt:
    '''
    Prompt used for generating code comments.
    '''
    def __init__(self):
        pass

    def code_comment_generator_prompt(self, code):
        prompt = f"""
        You are an expert code annotator. Your task is to add inline comments to the following code.
        IMPORTANT:
        1. DO NOT modify any part of the code. The code must remain exactly as provided, including any syntax errors.
        2. ONLY add inline comments using the appropriate comment syntax. Do NOT add any additional text or explanations outside the code.
        3. Your output should consist of the original code with inline comments inserted, with no extra headers or introductory text.
        Here is the code:
        ----------------
        {code}
        ----------------
        Please output the annotated code exactly as specified, with comments added inline to explain the functionality of each line of code.
        """
        return prompt

class OssInstGeneratorPrompt:
    '''
    The prompt for the oss inst generator.
    '''
    def __init__(self):
        pass
    
    def oss_inst_generator_prompt(self, code: str) -> str:
        return f'''
        You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions. Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in three distinct sections: **Problem Description**, **Analysis**, and **Solution**. 

        Guidelines for each section:  
        1. **Problem Description**: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.  
        2. **Analysis**: Offer a clear, logical breakdown of the problem, detailing the steps and thought processes required to solve it. This section should help understand the reasoning behind the solution.  
        3. **Solution**: Provide a comprehensive, correct solution that addresses the **Problem Description**. Ensure the solution is in the form of a self-contained script that does not rely on any assumptions or external details not provided in the **Problem Description**.  
        **Code Snippet** 
        {code}
        '''

class AnswerGeneratorPrompt:
    '''
    The prompt for the answer generator.
    '''
    def __init__(self):
        pass

    def Classic_COT_Prompt(question: str) -> str:
        """
        为给定数学题目生成系统提示信息
        """
        prompt = (
            r'''You are an intelligent chatbot designed for writing the answer of the given math question.
    Remember: DO NOT output anything else, only output the answer you make.
    Generate a solution of a given math problem strictly following this format:
    1. Identify key components of the problem
    2. Apply theorems/formulas with step-by-step derivation
    3. Perform calculations with intermediate verification
    4. Final answer in \boxed{} notation

    Format Requirements:
    - Prefix each step with "→" (use the actual arrow symbol, not its Unicode escape sequence)
    - Ensure all symbols and special characters are presented using LaTeX commands where appropriate (e.g., ≥ as \\geq, ÷ as \\div)

    Example Template:
    Problem: Find the minimum value of function f(x) = x³ - 3x² + 4 on interval [-1, 3]

    Solution:
    1. Find critical points:
    → f'(x) = 3x² - 6x
    → Set derivative to zero: 3x(x-2) = 0 ⇒ x=0, x=2

    2. Evaluate function at critical points and endpoints:
    → f(-1) = (-1)^3 - 3(-1)^2 + 4 = -1 -3 +4 = 0.0000
    → f(0) = 0³ - 3(0)² +4 = 4.0000
    → f(2) = 8 - 12 +4 = 0.0000
    → f(3) = 27 - 27 +4 = 4.0000

    3. Compare values:
    → Minimum occurs at x=-1 and x=2

    Verification:
    → Second derivative test: f''(x) = 6x-6
    → f''(-1) = -12 < 0 (local max)
    → f''(2) = 6 > 0 (local min)

    \boxed{0}

    Here is the given problem you need to solve:
    '''
        )
        return prompt + question + r'''Your response must directly start with "Solution:" without any preamble, After the answer is generated finish your response right away.'''
    

class QuestionSynthesisPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self,items, question):
        prompt = f"""
        Create a new reasonable and solvable math problem from the original problem by applying some of the following transformations(focus on all the transformations of "{items}"):

        1. Alter numerical values or expressions, ensuring the new problem remains solvable.
        2. Modify the problem type: introduce concepts like ratios or percentages, switch between derivatives and integrals, change the question from finding an area to finding a perimeter, etc.
        3. Contextualize the problem within a real-world scenario, such as incorporating various payment methods or deferred payments with interest.
        4. Add additional premises that require considering an extra factor separately in solving the problem.
        5. Increase the complexity of the problem by introducing multiple conditions that necessitate case-by-case analysis for a solution.

        Here is the problem from the user:
        {question}
        Write another problem inspired by this one.
        Not only change the problem scenario, but also try to create a new problem that requires another approach to solve.
        Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
        After the problem is generated finish your response right away.
        """
        return prompt
    
class QuestionCategoryPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self, question):
        prompt = f"""
        You are a classification assistant specialized in mathematics. Your task is to classify the given text into one primary category and one secondary category according to the following taxonomy. Do not output any extra explanation. Return only a JSON object with the keys "primary_category" and "secondary_category".

        Taxonomy:
        1. Foundations and Logic
        - 1.1 Mathematical Logic and Set Theory
        - 1.2 Basic Theory, Formalization, and History & Education

        2. Algebra and Number Theory
        - 2.1 Linear Algebra and Group Theory
        - 2.2 Ring Theory, Field Theory, and Polynomial Algebra
        - 2.3 Commutative Algebra and Homological/Categorical Methods
        - 2.4 Number Theory
        - 2.5 Algebraic Geometry

        3. Analysis and Differential Equations
        - 3.1 Real Analysis, Measure Theory, and Functional Analysis
        - 3.2 Complex Analysis and Special Functions
        - 3.3 Differential Equations and Dynamical Systems
        - 3.4 Integral Transforms, Integral Equations, and Difference Equations
        - 3.5 Harmonic Analysis

        4. Geometry and Topology
        - 4.1 Euclidean, Analytic, and Convex/Discrete Geometry
        - 4.2 Differential Geometry and Manifold Theory
        - 4.3 Topology and Algebraic Topology

        5. Probability, Statistics, and Discrete Mathematics
        - 5.1 Probability Theory and Stochastic Processes
        - 5.2 Mathematical Statistics
        - 5.3 Combinatorics and Graph Theory

        6. Applied and Computational Mathematics
        - 6.1 Numerical Analysis and Computational Methods
        - 6.2 Optimal Control, Variational Methods, and Optimization
        - 6.3 Operations Research and Game Theory
        - 6.4 Systems Theory and Control
        - 6.5 Computer Science and Algorithms
        - 6.6 Mathematical Physics and Engineering Mathematics
        - 6.7 Information and Communication
        - 6.8 Biomathematics

        7. Arithmetic
        - 7.1 Basic Arithmetic and Number Operations
        - 7.2 Word Problems and Real-Life Applications

        Classify the following text into one primary category and one secondary category based on the taxonomy above. The text is:
        {question}
        """
        return prompt

    
class QuestionDifficultyPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def question_synthesis_prompt(self, question):
        prompt = r"""
        # CONTEXT #
        I am a teacher, and I have some high-level olympiad math problems. 
        I want to evaluate the difficulty of these math problems. There are some references available regarding the difficulty of the problems:
        <difficulty reference>
        For reference, here are some sample problems from each of the difficulty levels 1-10:

        1: Jamie counted the number of edges of a cube, Jimmy counted the numbers of corners, and Judy counted the number of faces. They then added the three numbers. What was the resulting sum? (2003 AMC 8, Problem 1)

        1: How many integer values of $x$ satisfy $|x| < 3\pi$? (2021 Spring AMC 10B, Problem 1)

        1.5: A number is called flippy if its digits alternate between two distinct digits. For example, $2020$ and $37373$ are flippy, but $3883$ and $123123$ are not. How many five-digit flippy numbers are divisible by $15?$ (2020 AMC 8, Problem 19)

        2: A fair $6$-sided die is repeatedly rolled until an odd number appears. What is the probability that every even number appears at least once before the first occurrence of an odd number? (2021 Spring AMC 10B, Problem 18)

        2.5: $A$, $B$, $C$ are three piles of rocks. The mean weight of the rocks in $A$ is $40$ pounds, the mean weight of the rocks in $B$ is $50$ pounds, the mean weight of the rocks in the combined piles $A$ and $B$ is $43$ pounds, and the mean weight of the rocks in the combined piles $A$ and $C$ is $44$ pounds. What is the greatest possible integer value for the mean in pounds of the rocks in the combined piles $B$ and $C$? (2013 AMC 12A, Problem 16)

        3: Triangle $ABC$ with $AB=50$ and $AC=10$ has area $120$. Let $D$ be the midpoint of $\overline{AB}$, and let $E$ be the midpoint of $\overline{AC}$. The angle bisector of $\angle BAC$ intersects $\overline{DE}$ and $\overline{BC}$ at $F$ and $G$, respectively. What is the area of quadrilateral $FDBG$? (2018 AMC 10A, Problem 24)

        3.5: Find the number of integer values of $k$ in the closed interval $[-500,500]$ for which the equation $\log(kx)=2\log(x+2)$ has exactly one real solution. (2017 AIME II, Problem 7)

        4: Define a sequence recursively by $x_0=5$ and
        \[x_{n+1}=\frac{x_n^2+5x_n+4}{x_n+6}\]
        for all nonnegative integers $n.$ Let $m$ be the least positive integer such that
        \[x_m\leq 4+\frac{1}{2^{20}}.\]
        In which of the following intervals does $m$ lie?

        $\textbf{(A) } [9,26] \qquad\textbf{(B) } [27,80] \qquad\textbf{(C) } [81,242]\qquad\textbf{(D) } [243,728] \qquad\textbf{(E) } [729,\infty)$  
        (2019 AMC 10B, Problem 24 and 2019 AMC 12B, Problem 22)

        4.5: Find, with proof, all positive integers $n$ for which $2^n + 12^n + 2011^n$ is a perfect square. (USAJMO 2011/1)

        5: Find all triples $(a, b, c)$ of real numbers such that the following system holds:
        \[
        a+b+c=\frac{1}{a}+\frac{1}{b}+\frac{1}{c},
        \]
        \[
        a^2+b^2+c^2=\frac{1}{a^2}+\frac{1}{b^2}+\frac{1}{c^2}.
        \]
        (JBMO 2020/1)

        5.5: Triangle $ABC$ has $\angle BAC = 60^{\circ}$, $\angle CBA \leq 90^{\circ}$, $BC=1$, and $AC \geq AB$. Let $H$, $I$, and $O$ be the orthocenter, incenter, and circumcenter of $\triangle ABC$, respectively. Assume that the area of pentagon $BCOIH$ is the maximum possible. What is $\angle CBA$? (2011 AMC 12A, Problem 25)

        6: Let $\triangle ABC$ be an acute triangle with circumcircle $\omega,$ and let $H$ be the intersection of the altitudes of $\triangle ABC.$ Suppose the tangent to the circumcircle of $\triangle HBC$ at $H$ intersects $\omega$ at points $X$ and $Y$ with $HA=3,\ HX=2,$ and $HY=6.$ The area of $\triangle ABC$ can be written in the form $m\sqrt{n},$ where $m$ and $n$ are positive integers, and $n$ is not divisible by the square of any prime. Find $m+n.$ (2020 AIME I, Problem 15)

        6.5: Rectangles $BCC_1B_2,$ $CAA_1C_2,$ and $ABB_1A_2$ are erected outside an acute triangle $ABC.$ Suppose that
        \[\angle BC_1C+\angle CA_1A+\angle AB_1B=180^{\circ}.\]
        Prove that lines $B_1C_2,$ $C_1A_2,$ and $A_1B_2$ are concurrent. (USAMO 2021/1, USAJMO 2021/2)

        7: We say that a finite set $\mathcal{S}$ in the plane is balanced if, for any two different points $A$, $B$ in $\mathcal{S}$, there is a point $C$ in $\mathcal{S}$ such that $AC=BC$. We say that $\mathcal{S}$ is centre-free if for any three points $A$, $B$, $C$ in $\mathcal{S}$, there is no point $P$ in $\mathcal{S}$ such that $PA=PB=PC$.

        Show that for all integers $n\geq 3$, there exists a balanced set consisting of $n$ points.
        Determine all integers $n\geq 3$ for which there exists a balanced centre-free set consisting of $n$ points.
        (IMO 2015/1)

        7.5: Let $\mathbb{Z}$ be the set of integers. Find all functions $f : \mathbb{Z} \rightarrow \mathbb{Z}$ such that
        \[
        xf(2f(y)-x)+y^2f(2x-f(y))=\frac{f(x)^2}{x}+f(yf(y))
        \]
        for all $x, y \in \mathbb{Z}$ with $x \neq 0$. (USAMO 2014/2)

        8: For each positive integer $n$, the Bank of Cape Town issues coins of denomination $\frac1n$. Given a finite collection of such coins (of not necessarily different denominations) with total value at most $99+\frac{1}{2}$, prove that it is possible to split this collection into $100$ or fewer groups, such that each group has total value at most $1$. (IMO 2014/5)

        8.5: Let $I$ be the incentre of acute triangle $ABC$ with $AB\neq AC$. The incircle $\omega$ of $ABC$ is tangent to sides $BC, CA$, and $AB$ at $D, E,$ and $F$, respectively. The line through $D$ perpendicular to $EF$ meets $\omega$ at $R$. Line $AR$ meets $\omega$ again at $P$. The circumcircles of triangle $PCE$ and $PBF$ meet again at $Q$.

        Prove that lines $DI$ and $PQ$ meet on the line through $A$ perpendicular to $AI$. (IMO 2019/6)

        9: Let $k$ be a positive integer and let $S$ be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of $S$ around the circle such that the product of any two neighbors is of the form $x^2+x+k$ for some positive integer $x$. (IMO 2022/3)

        9.5: An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from $1$ to $10$.
        \[
        \begin{array}{ c@{\hspace{4pt}}c@{\hspace{4pt}} c@{\hspace{4pt}}c@{\hspace{2pt}}c@{\hspace{2pt}}c@{\hspace{4pt}}c }
        & & & 4 & & & \\
        & & 2 & & 6 & & \\
        & 5 & & 7 & & 1 & \\
        8 & & 3 & & 10 & & 9 \\
        \end{array}
        \]
        Does there exist an anti-Pascal triangle with $2018$ rows which contains every integer from $1$ to $1 + 2 + 3 + \dots + 2018$? (IMO 2018/3)

        10: Prove that there exists a positive constant $c$ such that the following statement is true: Consider an integer $n > 1$, and a set $\mathcal S$ of $n$ points in the plane such that the distance between any two different points in $\mathcal S$ is at least 1. It follows that there is a line $\ell$ separating $\mathcal S$ such that the distance from any point of $\mathcal S$ to $\ell$ is at least $cn^{-1/3}$.

        (A line $\ell$ separates a set of points S if some segment joining two points in $\mathcal S$ crosses $\ell$.) (IMO 2020/6)
        ## Some known difficulty ratings of the competitions.
        
        </difficulty reference>

        # OBJECTIVE #
        A. Summarize the math problem in a brief sentence, describing the concepts involved in the math problem.
        B. Based on the source of the given problem, as well as the difficulty of the problems referenced in these materials and the solution to the current problem, please provide 
        an overall difficulty score for the current problem. The score should be a number between 1 and 10, with increments of 0.5, and should align perfectly with the materials.
        # STYLE #
        Data report.
        # TONE #
        Professional, scientific.
        # AUDIENCE #
        Students. Enable them to better understand the difficulty of the math problems.
        # RESPONSE: MARKDOWN REPORT #
        ## Summarization
        [Summarize the math problem in a brief paragraph.]
        ## Difficulty
        [Rate the difficulty of the math problem and give the reason.]
        # ATTENTION #- Add "=== report over ===" at the end of the report.
        <example math problem>
        The problem requires finding the missing value in the equation

        \[
        \frac{1}{9}+\frac{1}{18}=\frac{1}{\square}.
        \]

        In other words, determine the number that should replace the square such that the sum of the fractions on the left equals the fraction on the right.
        </example math problem>
        ## Summarization
        The problem requires finding a value that makes the equation $\\frac{1}{9}+\\frac{1}{18}=\\frac{1}{\\square}$. 
        This involves adding two fractions and determining the equivalent fraction.
        ## Difficulty
        Rating: 1
        Reason: This problem is straightforward and primarily involves basic fraction addition, making it suitable for early middle school students. 
        === report over ===
        </example math problem>
        Let $\mathcal{P}$ be a convex polygon with $n$ sides, $n\ge3$. Any set of $n - 3$ diagonals of $\mathcal{P}$ that do not intersect in the interior of the polygon determine a triangulation of $\mathcal{P}$ into $n - 2$ triangles. If $\mathcal{P}$ is regular and there is a triangulation of $\mathcal{P}$ consisting of only isosceles triangles, find all the possible values of $n$. 
        </example math problem>
        ## Summarization
        The problem asks for the possible values of $n$ for a regular n-sided polygon that can be completely triangulated into isosceles triangles using non-intersecting diagonals. 
        The solution involves analyzing the properties of the diagonals forming isosceles triangles and deducing that $n$ can be expressed in terms of powers of 2.
        ## Difficulty
        Rating: 7
        Reason: The problem involves understanding properties of isosceles triangles in the context of polygon triangulation and requires critical reasoning to establish 
        relationships between the number of sides and powers of 2, making it more complex than typical undergraduate-level problems.
        === report over ===
        <math problem>
        [Question]: \n

        """

        return prompt + question

class TextSQLConsistencyPrompt:
    '''
    The prompt for the question synthesis.
    '''
    def __init__(self):
        pass

    def text_sql_consistency_prompt(self, question, sql, evidence):
        if evidence != "":
            prompt = f"""
            ## SQL Consistency Verification Task
            
            **Objective**: Given the question, evidence and SQL query, determine if the SQL query correctly implements the requirements specified in the natural language Question.
            
            **Evaluation Criteria**:
            1. The SQL should reflect key elements from the Question:
            2. You can refer to the content in evidence to determine if the SQL meets the requirements of the question
            3. Since you are not given the database schema, you can only analyze the SQL query and its relation to the Question and evidence.
            4. Do not judge as inconsistent just because of the database schema
            
            **Input**:
            Question: {question}
            Evidence: {evidence}
            SQL: {sql}
            
            **Required Output Format**:
            Analysis: <Brief technical analysis of the alignment between Question and SQL>
            Conclusion: <"YES" if consistent or uncertain, "NO" if definitely inconsistent> (No other text)
            
            **Example**:
            Analysis: The SQL query correctly implements the requirements of the Question, (may be more).
            Conclusion: <YES>
            
            **Important Notes**:
            - Respond ONLY with the specified format above
            - "YES" should be used when SQL implements Question OR when you're uncertain
            - "NO" should be used when SQL contradicts the Question
            - Be strict with logical requirements but lenient with syntax variations
            """
        else:
            prompt = f"""
            ## SQL Consistency Verification Task
            
            **Objective**: Given the question and SQL query, determine if the SQL query correctly implements the requirements specified in the natural language Question.
            
            **Evaluation Criteria**:
            1. The SQL should reflect key elements from the Question:
            2. You can refer to the content in evidence to determine if the SQL meets the requirements of the question
            3. Since you are not given the database schema, you can only analyze the SQL query and its relation to the Question and evidence.
            4. Do not judge as inconsistent just because of the database schema
            
            **Input**:
            Question: {question}
            SQL: {sql}
            
            **Required Output Format**:
            Analysis: <Brief technical analysis of the alignment between Question and SQL>
            Conclusion: <"YES" if consistent or uncertain, "NO" if definitely inconsistent> (No other text)
            
            **Example**:
            Analysis: The SQL query correctly implements the requirements of the Question, (may be more).
            Conclusion: <YES>
            
            **Important Notes**:
            - Respond ONLY with the specified format above
            - "YES" should be used when SQL implements Question OR when you're uncertain
            - "NO" should be used when SQL contradicts the Question
            - Be strict with logical requirements but lenient with syntax variations
            """
        return prompt
    
class QuestionRefinePrompt:
    def __init__(self):
        pass

    def question_refine_prompt(self, question):
        """Refine the question"""
        prompt = (
            "Analyze the following question and determine if it needs clarification:\n"
            f"ORIGINAL QUESTION: {question}\n"
            "Instructions:\n"
            "1. If the question is already perfectly clear, output: 'NO'\n"
            "2. If clarification would help, rewrite it to be more precise while:\n"
            "   - Preserving all original meaning\n"
            "   - Not adding/removing any factual content\n"
            "   - Only improving clarity of expression\n\n"
            "Format your response exactly as:\n"
            "```\n"
            "ANALYSIS: <brief explanation of why rewrite is/isn't needed>\n"
            "RESULT: <either 'NO' or the rewritten question>\n"
            "```"
        )
        
        return prompt
    
class ExtraKnowledgePrompt:
    def __init__(self):
        pass

    def extra_knowledge_prompt(self, question, sql, schema):
        prompt = (
            "Analyze whether answering this database question requires additional knowledge beyond the provided SQL and schema.\n"
            f"QUESTION: {question}\n"
            f"SQL QUERY: {sql}\n"
            f"TABLE SCHEMA:\n{schema}\n\n"
            "Consider:\n"
            "1. Are there domain terms not explained in the schema?\n"
            "2. Does the query rely on implicit business rules?\n"
            "3. Is special knowledge needed to interpret results?\n\n"
            "Respond ONLY in this exact format:\n"
            "RESULT: <knowledge> OR RESULT: NO\n"
            "Where <knowledge> is a concise explanation of required additional knowledge.\n"
            "If no extra knowledge is needed, respond with exactly 'RESULT: NO'."
        )
        return prompt


class FinalPromptGeneration:
    def __init__(self):
        pass

    def dial_sql_cot_prompt(self, question, sql, schema, evidence):
        prompt = (
            "/* Given the following database schema: */\n"
            f"{schema}\n\n"
            f"/* Answer the following: {question} */\n"
            "Let's think step by step ",
        )

        return prompt
    
    def dial_sql_non_cot_prompt(self, question, sql, schema, evidence):
        prompt = (
            "/* Given the following database schema: */\n"
            f"{schema}\n\n"
            f"/* Answer the following: {question} */\n"
            "SELECT ",
        )

        return prompt
    
    def omni_sql_cot_prompt(self, question, sql, schema, evidence):
        prompt = (
            "Task Overview:\n"
            "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.\n\n"
            "Database Engine:\n"
            "SQLite\n\n"
            "Database Schema:\n"
            f"{schema}\n"
            "This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.\n\n"
            "Question:\n"
            f"{evidence}\n"
            f"{question}\n\n"
            "Instructions:\n" 
            "- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
            "- The generated query should return all of the information asked in the question without any missing or extra information.\n"
            "- Before generating the final SQL query, please think through the steps of how to write the query.\n\n"
            "Output Format:\n"
            "In your answer, please enclose the generated SQL query in a code block:\n```sql\n-- Your SQL query\n```\n\n"
            "Take a deep breath and think step by step to find the correct SQL query.\n"
        )

        return prompt
    
    def omni_sql_non_cot_prompt(self, question, sql, schema, evidence):
        prompt = (
            "Task Overview:\n"
            "You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.\n\n"
            "Database Engine:\n"
            "SQLite\n\n"
            "Database Schema:\n"
            f"{schema}\n"
            "This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.\n\n"
            "Question:\n"
            f"{evidence}\n"
            f"{question}\n\n"
            # "Instructions:\n" 
            # "- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
            # "- The generated query should return all of the information asked in the question without any missing or extra information.\n"
            # "- Before generating the final SQL query, please think through the steps of how to write the query.\n\n"
            "Output Format:\n"
            "In your answer, please enclose the generated SQL query in a code block:\n```sql\n-- Your SQL query\n```\n\n"
            "Take a deep breath and think step by step to find the correct SQL query.\n"
        )

        return prompt


