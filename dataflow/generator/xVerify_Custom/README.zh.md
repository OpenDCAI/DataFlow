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

## 📘 介绍

xVerify 是一款基于预训练大语言模型微调而来的评估工具，专为具有唯一正确答案的客观题设计。它能从超长推理过程中精准提取最终答案，并通过智能等价性比较，高效识别不同形式的数学表达、LaTeX 与字符串表达及自然语言描述之间的等价关系。无论是数学题、选择题、分类题还是简答题，xVerify 都能提供精准、灵活的答案评估，极大提升评测效率和准确性。

## 🌟 主要特点
- 广泛的适用性：适用于各种不同的客观题评估场景。
    - 支持多种题型：数学题、多选题、分类题、简答题。
    - 支持超长推理：可以有效处理包含超长推理链的回答，识别超长推理中的最终答案。
    - 支持多种语言：主要支持中英文回答，并兼容其他语言。
- 强大的等价性判断：有效识别各类表达形式不同的等价表述。
    - 简单的答案转换识别：字母选项（a -> A）、希腊字母（alpha -> α）等
    - 多种数学表达形式的等价性判断：LaTex 格式（\frac{4}{5} -> 4/5）、科学计数法（1.34 x 10^3 -> 13400）、自然语言形式（one hundred and twenty-three -> 123）等，并且针对有缺失或者不同形式的 LaTex 格式也能正确处理。
    - 自然语言的等价性判断：针对简答题，可以有效识别 LLM 回答语句是否与正确答案的内容对齐。
    - 针对选择题，xVerify 可准确识别 LLM 回答中的选项内容或编号，而不仅限于匹配正确答案的编号。例如，若 LLM 仅提及正确选项的内容而未提供编号，xVerify 仍能对其进行匹配并判断正确性。
- 多规格模型可选：提供多个不同规格、不同基座的 xVerify 模型，规避基座模型的偏见。
    - 提供多个不同参数规模的 xVerify 模型，涵盖 0.5B 到 32B，可以基于可用计算资源灵活选择。
    - 提供多个不同基座模型微调而来的 xVerify 模型，包括 Qwen 2.5、Gemma 2、LLaMA 3.1、LLaMA 3.2、GLM 4、Phi-4 系列模型，可以基于需要评估的模型灵活选择，规避模型偏见。

##  🧪 评估示例
我们展示了一些具有代表性的示例：
- 具有复杂表达式的数学题示例：下面 4 个示例中 LLM 回答均包含较长的推理过程，设计复杂的数学计算与推理，最终答案的表达式也比较困难。此外，示例中 LLM 回答和正确答案中数学表达式所使用的形式不一致，例如前三个示例中，最终答案的表达式比较复杂，并且 LLM 回答中所使用表达形式与正确答案不一致，第 4 个示例中最终答案使用自然语言的形式，基于 RegEx 的方法在这里将完全失效。
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
- 简答题示例：简答题的正确答案为自然语言表述，因而需要识别 LLM 回答的整体含义，不能直接在回答中匹配正确答案的语句。下面示例的 LLM 回答没有完整给出正确答案的语句。
```json
{
    "index": 5,
    "question": "What is greater: the combined 2011 populations of Rennington (Northumberland), Lydbrook (Gloucestershire), Stow-on-the-Wold (Gloucestershire) and Witney (Oxfordshire), or the 2022 population of London?",
    "correct_answer": "The 2022 population of London",
    "llm_output": "To answer this question, we need to find the populations of the given locations and compare them to the population of London in 2022.\n\n1. Find the 2011 populations of Rennington (Northumberland), Lydbrook (Gloucestershire), Stow-on-the-Wold (Gloucestershire), and Witney (Oxfordshire).\n   - Rennington (Northumberland): approximately 190 residents\n   - Lydbrook (Gloucestershire): approximately 1,341 residents\n   - Stow-on-the-Wold (Gloucestershire): approximately 3,555 residents\n   - Witney (Oxfordshire): approximately 29,469 residents\n\n2. Calculate the combined population of these locations in 2011.\n   - Combined population: 190 + 1341 + 3555 + 29469 = 37,355\n\n3. Find the 2022 population of London.\n   - The 2022 population of London is approximately 9,000,000.\n\n4. Compare the combined population of the four locations to the population of London in 2022.\n   - The population of London (approximately 9,000,000) is greater than the combined population of the four locations (approximately 37,355).\n\nThe answer is The population of London is greater.",
    "xVerify_judgment": "Correct"
}
```
-  多选题示例：LLM 回答多选题时可能不提及选项对应的字母，所以需要评估方法能理解 LLM 的回答，并将选项内容与选项字母对应，才能进行判断。
```json
{
    "index": 6,
    "question": "Identity refers to one's  Answer Choices:  (A) Use of a particular coping strategy  (B) Control over one's own life story  (C) Physical appearance and overall personality  (D) Coherent sense of self",
    "correct_answer": "D",
    "llm_output": "The question asks about \"Identity,\" which is a complex concept related to how we perceive ourselves. \n\nLet's analyze the options:\n\n* **Use of a particular coping strategy:** This relates to how we deal with stress and challenges, not our sense of self.\n* **Control over one's own life story:** This touches on agency and self-determination, which are aspects of identity, but not the core definition.\n* **Physical appearance and overall personality:**  These are external factors that can influence how we perceive ourselves, but they don't fully encompass the essence of identity.\n* **Coherent sense of self:** This is the most accurate answer. Identity is about having a clear understanding of who you are, your values, beliefs, and how you see yourself in the world.",
    "xVerify_judgment": "Correct"
}
```
- 分类题示例
```json
{
    "index": 7,
    "question": "Where Is the Real Jayne Ann Krentz?: It was hard to believe that this novel was written by JAK. The prologue was the best part of the book. It started out with an explosion of character involvement and then the fire went out. She failed to give the reader a true insight into the main characters. I was hard pressed to understand why the two main characters were attracted to each other. I could only assume it was because they were trying to save a small company. There was no \"electricity\" between them. She had numerous secondary characters that never stood on their own. I read the entire book in one weekend, only because if I put the book down, I could not remember who was who. I hope the real JAK comes back to us soon.\nPlease identify the sentiment polarity of the sentence: positive or negative",
    "correct_answer": "negative",
    "llm_output": "To identify the sentiment polarity of the sentence, let's break it down:\n\n1. The reviewer states that it was \"hard to believe\" that the novel was written by JAK, implying that they were disappointed with the quality of the book.\n2. They mention that the prologue was the best part of the book, but this is a negative statement because it implies that the rest of the book was not as good.\n3. The reviewer criticizes the author for failing to give the reader a true insight into the main characters and for not making the characters' attraction to each other believable.\n4. They also mention that the secondary characters were not well-developed, which is another negative point.\n5. The reviewer states that they read the book in one weekend, but this is not a positive statement because it implies that they were trying to finish the book quickly rather than enjoying it.\n6. Finally, they express a hope that the \"real JAK\" will return, which implies that they are disappointed with the current quality of the author's work.\n\nOverall, the sentiment polarity of the sentence is negative.",
    "xVerify_judgment": "Correct"
}
```


## 🚀 快速开始
1. 克隆仓库
   * `git clone https://github.com/IAAR-Shanghai/xVerify.git && cd xVerify`
   
2. 准备环境
   * `conda create -n xverify python=3.10.13`
   * `conda activate xverify`
   
3. 安装 Python 依赖项
   * `pip install -r requirements.txt`
   
4. 模型部署
   * 在 Hugging Face 中下载合适的 xVerify 模型：
       
       | xVerify Model | HF Checkpoint                                                | Size | License         |
       | ------------- | ------------------------------------------------------------ | ---- | --------------- |
       | xVerify-0.5B-I  | [🤗 IAAR-Shanghai/xVerify-0.5B-I](https://huggingface.co/IAAR-Shanghai/xVerify-0.5B-I) | 0.5B   | CC-BY-NC-ND-4.0 |
       | xVerify-3B-Ia  | [🤗 IAAR-Shanghai/xVerify-3B-Ia](https://huggingface.co/IAAR-Shanghai/xVerify-3B-Ia) | 3B   | CC-BY-NC-ND-4.0 |
       | xVerify-8B-I  | [🤗 IAAR-Shanghai/xVerify-8B-I](https://huggingface.co/IAAR-Shanghai/xVerify-8B-I) | 8B   | CC-BY-NC-ND-4.0 |
       | xVerify-9B-C  | [🤗 IAAR-Shanghai/xVerify-9B-C](https://huggingface.co/IAAR-Shanghai/xVerify-9B-C) | 9B   | CC-BY-NC-ND-4.0 |
       
       > 💡 **Tip**
       >
       > 其中，xVerify-0.5B-I 以较小的参数规模表现优异，适合低成本部署；而 xVerify-9B-C 在综合表现上最佳，适合对性能有较高需求的用户。
       
   * 完成相应模型的本地部署，保证兼容 openai 请求接口，推荐基于 vLLM 进行部署（了解更多请参考 [vllm](https://github.com/vllm-project/vllm)）：
     
     ```bash
     # Basic deployment
     vllm serve --model ./models/your-merged-model --tensor-parallel-size 1
     # High-throughput configuration
     vllm serve --model ./models/your-merged-model --tensor-parallel-size 2 --max-model-len 8192
     ```
   
5. 数据准备
   * 可以直接使用 [eval_examples.json](src/xVerify/examples/eval_examples.json) 中的示例数据进行测试
   * 如果自行准备数据，请参照 [eval_examples.json](src/xVerify/examples/eval_examples.json) 中的数据格式准备数据，需要保证每个样本包含以下元素：
       * question: 问题文本
       * llm_output: LLM 针对问题的问答
       * correct_answer: 该问题的正确答案
        ```json
        {
            "question": "In which year did Fayaz A. Malik (an Indian pharmacologist, cancer biologist, and scientist) receive the Young Scientist of the Year from the Council of Scientific and Industrial Research?",
            "llm_output": "The year Fayaz A. Malik received the Young Scientist of the Year award from the Council of Scientific and Industrial Research was 2001.\n\nThe answer is 2001.",
            "correct_answer": "2009"
        }
        ```
   
6. 开始评估
   * 参考 [demo.ipynb](demo.ipynb) 进行评估，支持单样本评估和批量评估两种方式。
       * 🎯 单样本评估
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
       * 📚 批量评估
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

## 📊 实验结果
我们收集了目前常用于评估 LLM 回答正确性的评估框架和 judge model，并在四类客观题（字母选择题、数学题、简答题、分类题）上与 xVerify 进行了对比评估。其中，测试集中的样本与 xVerify 的训练集分布相同，即来自同一批的 LLM 在同一批评估集（客观题）上生成的回答，而泛化集则是采用不同的评估集询问 LLM 来生成样本，并且新增了多个新的 LLM 进行回答来生成样本，所以泛化集中的样本分布与训练集和测试集存在显著差异，其涵盖范围也更加广泛，可以有效检验 xVerify 的泛化性。
下表展示了测试集和泛化集上所有评估结果，"-" 表示评估方法在该类问题上不适用，每列最优值将被加粗，每列次优值将被标记下划线。
<ul>
    <li>测试集评估结果</li>
    <p align="center"><img src="./assets/test_results.png" alt=""></p>
    <li>泛化集评估结果</li>
    <p align="center"><img src="./assets/generalization_results.png" alt=""></p>
</ul>

## 📞 联系我们
如有任何问题、反馈或建议，请打开 GitHub Issue。您可以通过 [GitHub Issues](https://github.com/IAAR-Shanghai/xVerify/issues) 联系我们。

## 🔗 引用
```
@misc{xverify_25_github,
    author = {Ding Chen and Qingchen Yu and Bo Tang and Feiyu Xiong and Zhiyu Li},
    title = {xVerify: Efficient Answer Verifier for Large Language Model Evaluations},
    url = {https://github.com/IAAR-Shanghai/xVerify},
    year={2025}
}
```
