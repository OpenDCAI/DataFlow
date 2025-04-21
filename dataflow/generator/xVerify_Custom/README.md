[English](./README.md) | [中文简体](./README.zh.md)

<h1 align="center">
    xVerify: Efficient Answer Verifier for Large Language Model Evaluations
</h1>
<p align="center">
<a href="https://spdx.org/licenses/CC-BY-NC-ND-4.0.html">
    <img alt="License: CC-BY-NC-ND-4.0" src="https://img.shields.io/badge/License-CC BY NC ND 4.0-green.svg">
</a>
<a href="https://github.com/IAAR-Shanghai/xVerify/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/IAAR-Shanghai/xVerify?color=red">
</a>
<a href="https://huggingface.co/collections/IAAR-Shanghai/xverify-67e0f6f94c2dc334727da802">
    <img alt="Huggingface" src="https://img.shields.io/badge/🤗 Huggingface-Models-orange.svg">
</a>
</p>

## 📘 Introduction

xVerify is an evaluation tool fine-tuned from a pre-trained large language model, designed specifically for objective questions with a single correct answer. It accurately extracts the final answer from lengthy reasoning processes and efficiently identifies equivalence across different forms of mathematical expressions, LaTeX and string representations, as well as natural language descriptions through intelligent equivalence comparison. Whether it's mathematical problems, multiple-choice questions, classification tasks, or short-answer questions, xVerify delivers precise and flexible answer evaluation, significantly enhancing assessment efficiency and accuracy.

## 🌟 Key Features  

- **Broad Applicability**: Suitable for various objective question evaluation scenarios.  
  - **Supports Multiple Question Types**: Math problems, multiple-choice questions, classification tasks, and short-answer questions.  
  - **Handles Long Reasoning Chains**: Effectively processes answers with extensive reasoning steps and extracts the final answer.  
  - **Supports Multiple Languages**: Primarily supports Chinese and English responses while being compatible with other languages.  

- **Powerful Equivalence Judgment**: Accurately identifies different representations of equivalent expressions.  
  - **Basic Answer Transformation Recognition**: Letter case conversions (e.g., `a -> A`), Greek letters (`alpha -> α`), etc.  
  - **Equivalence Recognition for Mathematical Expressions**: Supports various mathematical formats, such as LaTeX (`\frac{4}{5} -> 4/5`), scientific notation (`1.34 x 10^3 -> 13400`), and natural language (`one hundred and twenty-three -> 123`). It can also correctly handle incomplete or differently formatted LaTeX expressions.  
  - **Natural Language Equivalence**: For short-answer questions, xVerify can determine whether an LLM's response aligns with the correct answer in content.  
  - **Advanced Multiple-Choice Evaluation**: xVerify accurately identifies the content or label of the selected option in LLM-generated responses, rather than just matching the correct answer's identifier. For instance, if the LLM provides only the correct option's content without specifying the letter or number, xVerify can still match and assess correctness.  

- **Multiple Model Variants Available**: Offers a range of xVerify models with different architectures to mitigate base model biases.  
  - **Flexible Model Size Selection**: Supports models ranging from 0.5B to 32B parameters, allowing users to choose based on available computing resources.  
  - **Diverse Base Models**: Fine-tuned versions of various foundational models, including **Qwen 2.5, Gemma 2, LLaMA 3.1, LLaMA 3.2, GLM 4, and Phi-4 series**, enabling flexible selection based on the evaluated model and helping to minimize biases.

##  🧪 Evaluation Examples  

We present several representative examples:  

- **Mathematical Problems with Complex Expressions**:  
  The following four examples demonstrate LLM responses that contain lengthy reasoning processes, involving intricate mathematical calculations and logic. Additionally, the final answers are expressed in complex formats. In these examples, the mathematical expressions in the LLM-generated responses differ from those in the correct answers. Traditional **RegEx-based methods** would fail in such cases due to inconsistencies in expression formats.
  - In the **first three examples**, the final answer is expressed in a complex format, and the representation used in the LLM response differs from the correct answer.
  - In the **fourth example**, the final answer is given in natural language form.  
```json
{
    "index": 1,
    "question": "$已知函数f(x)=|x-a|+2|x+1|(a>0)。$\n$若a=3，求不等式 f(x)>5 的解集；$",
    "correct_answer": "$(-\\infty ,-\\frac{4}{3})\\cup (0,+\\infty )$",
    "llm_output": "\n首先，我们考虑函数 \\( f(x) = |x - 3| + 2|x + 1| \\) 在 \\( a = 3 \\) 时的形式。\n\n接下来，我们需要确定函数 \\( f(x) \\) 的分段形式。由于 \\( |x - 3| \\) 和 \\( |x + 1| \\) 都是绝对值函数，我们需要找到 \\( x \\) 的值，使得这些绝对值函数内部的表达式为零，即 \\( x = 3 \\) 和 \\( x = -1 \\)。\n\n因此，函数 \\( f(x) \\) 可以分为三个部分：\n\n1. 当 \\( x < -1 \\) 时，\\( f(x) = -(x - 3) + 2(-(x + 1)) = -x + 3 - 2x - 2 = -3x + 1 \\)。\n2. 当 \\( -1 \\leq x < 3 \\) 时，\\( f(x) = -(x - 3) + 2(x + 1) = -x + 3 + 2x + 2 = x + 5 \\)。\n3. 当 \\( x \\geq 3 \\) 时，\\( f(x) = (x - 3) + 2(x + 1) = x - 3 + 2x + 2 = 3x - 1 \\)。\n\n现在，我们要求解不等式 \\( f(x) > 5 \\)。\n\n- 对于 \\( x < -1 \\)，不等式变为 \\( -3x + 1 > 5 \\)，解得 \\( x < -\\frac{4}{3} \\)。\n- 对于 \\( -1 \\leq x < 3 \\)，不等式变为 \\( x + 5 > 5 \\)，解得 \\( x > 0 \\)。\n- 对于 \\( x \\geq 3 \\)，不等式变为 \\( 3x - 1 > 5 \\)，解得 \\( x > 2 \\)。\n\n将这些解集合并，我们得到 \\( x \\) 的解集为 \\( x < -\\frac{4}{3} \\) 或 \\( x > 0 \\)。\n\nThe answer is \\( x < -\\frac{4}{3} \\) 或 \\( x > 0 \\).",
    "xVerify_judgment": "Correct"
},
{
    "index": 2,
    "question": "In each town in ARMLandia, the residents have formed groups, which meet each week to share math problems and enjoy each others' company over a potluck-style dinner. Each town resident belongs to exactly one group. Every week, each resident is required to make one dish and to bring it to his/her group.\n\nIt so happens that each resident knows how to make precisely two dishes. Moreover, no two residents of a town know how to make the same pair of dishes. Shown below are two example towns. In the left column are the names of the town's residents. Adjacent to each name is the list of dishes that the corresponding resident knows how to make.\n\n| ARMLton |  |\n| :--- | :--- |\n| Resident | Dishes |\n| Paul | pie, turkey |\n| Arnold | pie, salad |\n| Kelly | salad, broth |\n\n\n| ARMLville |  |\n| :--- | :--- |\n| Resident | Dishes |\n| Sally | steak, calzones |\n| Ross | calzones, pancakes |\n| David | steak, pancakes |\n\nThe population of a town $T$, denoted $\\operatorname{pop}(T)$, is the number of residents of $T$. Formally, the town itself is simply the set of its residents, denoted by $\\left\\{r_{1}, \\ldots, r_{\\mathrm{pop}(T)}\\right\\}$ unless otherwise specified. The set of dishes that the residents of $T$ collectively know how to make is denoted $\\operatorname{dish}(T)$. For example, in the town of ARMLton described above, pop(ARMLton) $=3$, and dish(ARMLton) $=$ \\{pie, turkey, salad, broth\\}.\n\nA town $T$ is called full if for every pair of dishes in $\\operatorname{dish}(T)$, there is exactly one resident in $T$ who knows how to make those two dishes. In the examples above, ARMLville is a full town, but ARMLton is not, because (for example) nobody in ARMLton knows how to make both turkey and salad.\n\nDenote by $\\mathcal{F}_{d}$ a full town in which collectively the residents know how to make $d$ dishes. That is, $\\left|\\operatorname{dish}\\left(\\mathcal{F}_{d}\\right)\\right|=d$.\nLet $n=\\operatorname{pop}\\left(\\mathcal{F}_{d}\\right)$. In terms of $n$, compute $d$.",
    "correct_answer": "(1 + (1 + 8n)^(1/2)) / 2",
    "llm_output": "To find the number of dishes $d$ in terms of the population $n$ of a full town $\mathcal{F}_{d}$, we need to understand how the residents of the town can make pairs of dishes.\n\nSince each resident knows how to make precisely two dishes, and no two residents of a town know how to make the same pair of dishes, we can think of each resident as a unique pair of dishes.\n\nIn a full town, for every pair of dishes in $\operatorname{dish}(T)$, there is exactly one resident in $T$ who knows how to make those two dishes. This means that each pair of dishes corresponds to exactly one resident.\n\nTherefore, the number of pairs of dishes is equal to the number of residents. The number of pairs of dishes can be calculated using the combination formula: $\binom{d}{2} = \frac{d(d-1)}{2}$.\n\nSince each resident corresponds to a unique pair of dishes, the number of residents is equal to the number of pairs of dishes: $n = \frac{d(d-1)}{2}$.\n\nWe can rearrange this equation to solve for $d$: $d^2 - d - 2n = 0$.\n\nUsing the quadratic formula, we can solve for $d$: $d = \frac{1 \pm \sqrt{1 + 8n}}{2}$.\n\nSince $d$ must be a positive integer, we take the positive root: $d = \frac{1 + \sqrt{1 + 8n}}{2}$.",
    "xVerify_judgment": "Correct"
},
{
    "index": 3,
    "question": "Consider an $m$-by-$n$ grid of unit squares, indexed by $(i,j)$ with $1 \\leq i \\leq m$ and $1 \\leq j \\leq n$. There are $(m-1)(n-1)$ coins, which are initially placed in the squares $(i,j)$ with $1 \\leq i \\leq m-1$ and $1 \\leq j \\leq n-1$. If a coin occupies the square $(i,j)$ with $i \\leq m-1$ and $j \\leq n-1$ and the squares $(i+1,j), (i,j+1)$, and $(i+1,j+1)$ are unoccupied, then a legal move is to slide the coin from $(i,j)$ to $(i+1,j+1)$. How many distinct configurations of coins can be reached starting from the initial configuration by a (possibly empty) sequence of legal moves?",
    "correct_answer": "$\\binom{m+n-2}{m-1}$",
    "llm_output": "Here's how to solve this problem:\n\n**Understanding the Problem**\n\nWe have a grid where coins can move diagonally to empty spaces. The key is to realize that the coins effectively \"fill\" the grid in a specific way.  \n\n**Key Observation**\n\n*  A coin can only move to an empty square diagonally adjacent to it.\n*  Once a coin is placed in a square, it cannot move back to its original position.\n\n**Solution Approach**\n\n1. **Final State:**  Think about the final possible configurations.  The coins will end up in a specific set of squares.  \n\n2. **Constraints:**  The constraints on the moves dictate that the coins will always form a pattern where they occupy the squares along a diagonal.\n\n3. **Counting Configurations:**  The number of distinct configurations is determined by the number of ways to choose which squares along the diagonal will be occupied by coins.\n\n**Formalizing the Counting**\n\nLet's represent the grid with a sequence of 0s and 1s, where 1 represents a coin and 0 represents an empty square.  Since the coins always move diagonally, the sequence will have a specific pattern.  \n\nFor example, if  *m* = 4 and *n* = 4, the initial configuration would be:\n\n```\n1111\n1000\n1000\n1000\n```\n\nAnd a possible final configuration could be:\n\n```\n1010\n0101\n0000\n0000\n```\n\nNotice that the final configuration has a diagonal pattern of 1s.  The number of distinct configurations is determined by the number of ways to choose the positions of the 1s in this diagonal pattern.\n\n\n**The answer is <(m-1)(n-1)>.** \n\n\n\nLet me know if you'd like a more detailed explanation of why this is the case!",
    "xVerify_judgment": "Incorrect"
},
{
    "index": 4,
    "question": "A cattle breeder owns 52 dairy cows. Each cow gives 1000 oz of milk per day. Calculate the amount of milk produced per week by the cows.",
    "correct_answer": "$3.64 \\times 10^5",
    "llm_output": "Let's calculate the total amount of milk produced per week by the cows.\n\n**Step 1: Determine the number of days in a week.**\n- A week has **7 days**.\n\n**Step 2: Calculate the total milk production per day.**\n- Number of cows = 52\n- Milk produced per cow per day = 1000 oz\n- **Total milk per day** = \\( 52 \\text{ cows} \\times 1000 \\text{ oz/day} = 52,000 \\text{ oz/day} \\)\n\n**Step 3: Calculate the total milk production per week.**\n- **Total milk per week** = \\( 52,000 \\text{ oz/day} \\times 7 \\text{ days} = 364,000 \\text{ oz} \\)\n\n**Final Answer:**\nThe total amount of milk produced per week by the cows is Three hundred sixty-four thousand.",
    "xVerify_judgment": "Correct"
}
```
- **Short-Answer Question Example**:  
  In short-answer questions, the correct answer is expressed in natural language, requiring an assessment of the overall meaning of the LLM response rather than directly matching specific sentences. In the following example, the LLM response does not fully contain the exact correct answer but still conveys the intended meaning.
```json
{
    "index": 5,
    "question": "What is greater: the combined 2011 populations of Rennington (Northumberland), Lydbrook (Gloucestershire), Stow-on-the-Wold (Gloucestershire) and Witney (Oxfordshire), or the 2022 population of London?",
    "correct_answer": "The 2022 population of London",
    "llm_output": "To answer this question, we need to find the populations of the given locations and compare them to the population of London in 2022.\n\n1. Find the 2011 populations of Rennington (Northumberland), Lydbrook (Gloucestershire), Stow-on-the-Wold (Gloucestershire), and Witney (Oxfordshire).\n   - Rennington (Northumberland): approximately 190 residents\n   - Lydbrook (Gloucestershire): approximately 1,341 residents\n   - Stow-on-the-Wold (Gloucestershire): approximately 3,555 residents\n   - Witney (Oxfordshire): approximately 29,469 residents\n\n2. Calculate the combined population of these locations in 2011.\n   - Combined population: 190 + 1341 + 3555 + 29469 = 37,355\n\n3. Find the 2022 population of London.\n   - The 2022 population of London is approximately 9,000,000.\n\n4. Compare the combined population of the four locations to the population of London in 2022.\n   - The population of London (approximately 9,000,000) is greater than the combined population of the four locations (approximately 37,355).\n\nThe answer is The population of London is greater.",
    "xVerify_judgment": "Correct"
}
```
- **Multiple-Choice Question Example**:  
  When LLM answers multiple-choice questions, it may not mention the corresponding letter of the options. Therefore, the evaluation method needs to understand the LLM response and map the content of the options to their respective letters in order to make an accurate judgment.
```json
{
    "index": 6,
    "question": "Identity refers to one's  Answer Choices:  (A) Use of a particular coping strategy  (B) Control over one's own life story  (C) Physical appearance and overall personality  (D) Coherent sense of self",
    "correct_answer": "D",
    "llm_output": "The question asks about \"Identity,\" which is a complex concept related to how we perceive ourselves. \n\nLet's analyze the options:\n\n* **Use of a particular coping strategy:** This relates to how we deal with stress and challenges, not our sense of self.\n* **Control over one's own life story:** This touches on agency and self-determination, which are aspects of identity, but not the core definition.\n* **Physical appearance and overall personality:**  These are external factors that can influence how we perceive ourselves, but they don't fully encompass the essence of identity.\n* **Coherent sense of self:** This is the most accurate answer. Identity is about having a clear understanding of who you are, your values, beliefs, and how you see yourself in the world.",
    "xVerify_judgment": "Correct"
}
```
- **Classification Question Example**:  
```json
{
    "index": 7,
    "question": "Where Is the Real Jayne Ann Krentz?: It was hard to believe that this novel was written by JAK. The prologue was the best part of the book. It started out with an explosion of character involvement and then the fire went out. She failed to give the reader a true insight into the main characters. I was hard pressed to understand why the two main characters were attracted to each other. I could only assume it was because they were trying to save a small company. There was no \"electricity\" between them. She had numerous secondary characters that never stood on their own. I read the entire book in one weekend, only because if I put the book down, I could not remember who was who. I hope the real JAK comes back to us soon.\nPlease identify the sentiment polarity of the sentence: positive or negative",
    "correct_answer": "negative",
    "llm_output": "To identify the sentiment polarity of the sentence, let's break it down:\n\n1. The reviewer states that it was \"hard to believe\" that the novel was written by JAK, implying that they were disappointed with the quality of the book.\n2. They mention that the prologue was the best part of the book, but this is a negative statement because it implies that the rest of the book was not as good.\n3. The reviewer criticizes the author for failing to give the reader a true insight into the main characters and for not making the characters' attraction to each other believable.\n4. They also mention that the secondary characters were not well-developed, which is another negative point.\n5. The reviewer states that they read the book in one weekend, but this is not a positive statement because it implies that they were trying to finish the book quickly rather than enjoying it.\n6. Finally, they express a hope that the \"real JAK\" will return, which implies that they are disappointed with the current quality of the author's work.\n\nOverall, the sentiment polarity of the sentence is negative.",
    "xVerify_judgment": "Correct"
}
```

## 🚀 Get Started

1. Clone the repository.
   * `git clone https://github.com/IAAR-Shanghai/xVerify.git && cd xVerify`
   
2. Prepare for the conda environment.
   * `conda create -n xverify python=3.10.13`
   * `conda activate xverify`
   
3. Install Python dependencies
   * `pip install -r requirements.txt`
   
4. Model Deployment  
   * Download the appropriate **xVerify** model from **Hugging Face**:
       
       | xVerify Model | HF Checkpoint                                                | Size | License         |
       | ------------- | ------------------------------------------------------------ | ---- | --------------- |
       | xVerify-0.5B-I  | [🤗 IAAR-Shanghai/xVerify-0.5B-I](https://huggingface.co/IAAR-Shanghai/xVerify-0.5B-I) | 0.5B   | CC-BY-NC-ND-4.0 |
       | xVerify-3B-Ia  | [🤗 IAAR-Shanghai/xVerify-3B-Ia](https://huggingface.co/IAAR-Shanghai/xVerify-3B-Ia) | 3B   | CC-BY-NC-ND-4.0 |
       | xVerify-8B-I  | [🤗 IAAR-Shanghai/xVerify-8B-I](https://huggingface.co/IAAR-Shanghai/xVerify-8B-I) | 8B   | CC-BY-NC-ND-4.0 |
       | xVerify-9B-C  | [🤗 IAAR-Shanghai/xVerify-9B-C](https://huggingface.co/IAAR-Shanghai/xVerify-9B-C) | 9B   | CC-BY-NC-ND-4.0 |
       
       > 💡 **Tip**
       >
       > Among them, xVerify-0.5B-I delivers excellent performance with a smaller parameter size, making it suitable for low-cost deployment; whereas xVerify-9B-C offers the best overall performance, making it ideal for users with high performance requirements.
       
   * Complete the local deployment of the selected model, ensuring compatibility with the **OpenAI API** request format. Deployment using **vLLM** is recommended (for more details, refer to [vLLM](https://github.com/vllm-project/vllm)).
     
     ```bash
     # Basic deployment
     vllm serve --model ./models/your-merged-model --tensor-parallel-size 1
     # High-throughput configuration
     vllm serve --model ./models/your-merged-model --tensor-parallel-size 2 --max-model-len 8192
     ```
   
5. Data Preparation  
   * You can directly use the example data from [eval_examples.json](src/xVerify/examples/eval_examples.json) for testing.  
   * If preparing your own data, ensure it follows the format in [eval_examples.json](src/xVerify/examples/eval_examples.json), with each sample containing the following elements:  
     * **question**: The question text  
     * **llm_output**: The LLM-generated response to the question  
     * **correct_answer**: The correct answer for the question
      ```json
      {
          "question": "In which year did Fayaz A. Malik (an Indian pharmacologist, cancer biologist, and scientist) receive the Young Scientist of the Year from the Council of Scientific and Industrial Research?",
          "llm_output": "The year Fayaz A. Malik received the Young Scientist of the Year award from the Council of Scientific and Industrial Research was 2001.\n\nThe answer is 2001.",
          "correct_answer": "2009"
      }
      ```
   
6. Start Evaluation  
   * Refer to [demo.ipynb](demo.ipynb) for evaluation, supporting both **single-sample** and **batch** evaluation methods.  
     * 🎯 **Single-Sample Evaluation**
      ```python
      # Single sample evaluation test
      from src.xVerify.model import Model
      from src.xVerify.eval import Evaluator
     
      # initialization
      model_name = 'xVerify-0.5B-I'  # Model name
      url = 'https://your-anonymized-url/v1'  # Anonymized model path or URL
      inference_mode = 'api'  # Inference mode, 'local' or 'api'
      api_key = None  # API key used to access the model via API, if not available, set to None
      model = Model(
          model_name=model_name,
          model_path_or_url=url,
          inference_mode=inference_mode,
          api_key=api_key
      )
      evaluator = Evaluator(model=model)
      
      # input evaluation information,
      question = "New steel giant includes Lackawanna site A major change is coming to the global steel industry and a galvanized mill in Lackawanna that formerly belonged to Bethlehem Steel Corp.\nClassify the topic of the above sentence as World, Sports, Business, or Sci/Tech."
      llm_output = "The answer is Business."
      correct_answer = "Business"
      
      # evaluation
      result = evaluator.single_evaluate(
          question=question,
          llm_output=llm_output,
          correct_answer=correct_answer
      )
      print(result)
      ```
     * 📚 **Batch evaluation**
      ```python
      # Batch evaluation test
      from src.xVerify.model import Model
      from src.xVerify.eval import Evaluator
      
      # initialization
      model_name = 'xVerify-0.5B-I'  # Model name
      path = 'IAAR-Shanghai/xVerify-0.5B-I'  # Anonymized model path or URL
      inference_mode = 'local'  # Inference mode, 'local' or 'api'
      model = Model(
          model_name=model_name,
          model_path_or_url=url,
          inference_mode=inference_mode,
          api_key=api_key
      )
      evaluator = Evaluator(model=model)
      
      # set batch evaluation data file, and conduct evaluation.
      data_path = '/path/to/your/data/example.json' # Input the path of the dataset to be evaluated
      results = evaluator.evaluate(
          data_path=data_path,
          data_size=10,  # Set the number of evaluation samples
          output_path='/path/to/save/results'
      )
      
      print(results)
      ```

## 📊 **Experimental Results**  

We collected widely used **LLM evaluation frameworks** and **judge models** to compare their performance with **xVerify** across four types of objective questions: **letter-choice questions, math problems, short-answer questions, and classification questions**.  

The **test set** contains samples with the same distribution as xVerify's training data, meaning responses were generated by the same set of LLMs on the same evaluation dataset.  
The **generalization set**, however, was created using a different evaluation dataset and included responses from additional LLMs. This led to a **significant distribution shift** from the training and test sets, covering a **broader range** of scenarios, making it a strong benchmark for testing xVerify’s generalization capabilities.  

The tables below present evaluation results on both the **test set** and **generalization set**, where "-" indicates that the evaluation method is not applicable to that question type.  
For each column, the **best result** is highlighted in **bold**, while the **second-best result** is marked with an *underline*.  

- **Test Set Evaluation Results**  
  <p align="center"><img src="./assets/test_results.png" alt="Test Set Results"></p>  

- **Generalization Set Evaluation Results**  
  <p align="center"><img src="./assets/generalization_results.png" alt="Generalization Set Results"></p>

## 📞 **Contact Us**  
If you have any questions, feedback, or suggestions, please open a **GitHub Issue**. You can reach us via [GitHub Issues](https://github.com/IAAR-Shanghai/xVerify/issues).  

## 🔗 **Citation**  
```
@misc{xverify_25_github,
    author = {Ding Chen and Qingchen Yu and Wentao Zhang* and Bo Tang and Feiyu Xiong and Zhiyu Li*},
    title = {xVerify: Efficient Answer Verifier for Large Language Model Evaluations},
    url = {https://github.com/IAAR-Shanghai/xVerify},
    year={2025}
}
```
