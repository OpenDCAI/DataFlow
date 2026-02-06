from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class VQAExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, example_title, subject: str = "math", interleaved=True) -> str:
        PROMPT = ""
        if interleaved:
          PROMPT = f"""
        You are an expert in {subject} competition. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All {subject} problems whose text begins on page_n and the answers with solutions to those problems.
2. If the problem or answer is not complete (because they continue onto page_n+1), omit them. If the problem is complete but the solution not, omit both the problem and the solution. DO NOT INCLUDE INCOMPLETE PROBLEMS OR ANSWERS. However, if only the solution is incomplete, you may still include the question and the short answer if they are complete and leave the solution empty.
3. Normally, a box at the beginning of a page with no problem number (such as "1.1", "例 1", "example 1", "解", "solution", "答案", "answer") is the continuation of the problem or solution from the previous page, even if it appears to be an independent paragraph. Omit them.
4. The chapter information as it appears on page_n. YOU MUST INCLUDE ALL TITLES APPEARING ON THE PAGE, EVEN IF NO QUESTIONS OR ANSWERS ARE PRESENT UNDER THAT TITLE.
"""
        else:
          PROMPT = f"""
        You are an expert in {subject} competition. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All {subject} problems whose text appears on page_n. In the provided page, there will be all questions or all answers with solutions, but not mixed.
2. If the problem or answer is not complete (because they continue onto page_n+1), omit them. DO NOT INCLUDE INCOMPLETE QUESTIONS OR ANSWERS. However, if only the solution is incomplete, you may still include the question and the short answer (sometimes only a letter or number) if they are complete and leave the solution empty.
3. Normally, a box at the beginning of a page with no problem number (such as "1.1", "例 1", "example 1", "解", "solution", "答案", "answer") is the continuation of the problem or solution from the previous page, even if it appears to be an independent paragraph. Omit them.
4. The chapter information as it appears on page_n. YOU MUST INCLUDE ALL TITLES APPEARING ON THE PAGE, EVEN IF NO QUESTIONS OR ANSWERS ARE PRESENT UNDER THAT TITLE.
"""
        PROMPT +=f"""
When provided two column pages, you should read from **left to right**, top to bottom. Also output the extracted content from **left to right**, top to bottom.
Strict extraction rules:
** About questions and answers/solutions **
- If you think the page is not the main text page, such as a cover page, catalog page, header/footer only, etc., output `<empty></empty>`.
- Preserve each problem’s original label/number, such as "例1", "Example 3", "习题1", "11". Do not include the period after the number. Use Arabic numerals only. For example, if the label is "例一", convert it to "例1". If the label is "IV", convert it to "4". If the full label is "三、16", keep only "16".
- If there are multiple sub-questions under one main question, always put them together in the same `<qa_pair>`…`</qa_pair>` block.
- If a question and its answer/solution are contiguous on page_n, wrap them together as a single `<qa_pair>`…`</qa_pair>` block, e.g.:
  `<qa_pair><label>例1</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>`
- If only questions or only answers with solutions appear on page_n, wrap each question or answer with solution in a `<qa_pair>`…`</qa_pair>` block with the missing part left empty. For example, if only questions appear:
  `<qa_pair><label>例1</label><question>…</question><answer></answer><solution></solution></qa_pair>`
- If multiple questions and solutions appear on page_n, wrap each question/solution pair in its own `<qa_pair>`…`</qa_pair>` block.
- Sometimes a short answer may appear before the full solution. If you do not see the full solution on page_n, only extract the short answer and leave the solution empty.
** About chapter/section titles **
- Enclose the output in a `<chapter>`…`</chapter>` block, where <title>MAIN_TITLE</title> is the chapter title or section title appearing on page_n.
- There could be multiple `<chapter>`…`</chapter>` blocks if multiple chapters/sections appear on page_n.
- Extract chapter titles only, and with no prefix number. For example, "{example_title}", please strictly follow this example title. If you see multiple titles piled together, use the one at the bottom only.  
- **Do not keep subtitles. Any titles followed by a question/answer whose label is not 1 should be considered a subtitle. DO NOT EXTRACT THEM.**
- If you encounter a title with no problems or answers on the page, still extract the title within the `<chapter>`…`</chapter>` block, with an empty qa_pair block with label 0 `<qa_pair><label>0</label><question></question><answer></answer><solution></solution></qa_pair>`. Normally this will happen when the title is at the end of the page.
- Do not use nested titles.
- Leave the title blank if there is no chapter title of the questions. 
- Sometimes the chapter title may not appear at the beginning of the page. You should leave the title of all the qa pairs before the chapter title **blank**, but the title of the qa pairs after the chapter title should use this chapter title.
** About text and figures/diagrams **
- For problem and answer/solution text, output exactly what appears (no translation). Render all mathematical expressions in LaTeX.
- Whenever the question or answer/solution refers to a figure or diagram, record it with `<pic>tagA:boxB</pic>`, such as `<pic>tag5:box7</pic>`. tagA:boxB is labeled (in exactly the same format) in the image beside the figure or diagram in RED color. Be careful that the original caption of the book may also exist, but usually in format A.B (normally in black color). Do NOT use the original caption in the book!!! Additionally, the figure/diagram may be surrounded by multiple labels (some from other boxes), be careful to pick the correct one. The correct one will be at the upper left of the figure/diagram. If you are not sure, you are free to put multiple labels, e.g. `<pic>tag5:box7</pic> <pic>tag5:box8</pic>`. NEVER leave it blank or make up a label!
- You should always put the `<pic>...</pic>` tag at the exact position where the figure/diagram is referenced in the text. If there are multiple references, put multiple tags at the correct positions.

If no qualifying content is found, output:
<empty></empty>

Output format (all tags run together, no extra whitespace or newlines except between entries):
<chapter><title>MAIN_TITLE</title>
<qa_pair><label>…</label><question>QUESTION_TEXT<pic>…</pic>…</question>
<answer>ANSWER_TEXT<pic>…</pic>…</answer><solution>SOLUTION_TEXT</solution></qa_pair>
<qa_pair><label>…</label><question>QUESTION_TEXT<pic>…</pic>…</question>
<answer>ANSWER_TEXT<pic>…</pic>…</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>MAIN_TITLE</title>
<qa_pair><label>…</label><question>QUESTION_TEXT<pic>…</pic>…</question>
<answer>ANSWER_TEXT<pic>…</pic>…</answer><solution>SOLUTION_TEXT</solution></qa_pair>
</chapter>


Example:
<chapter><title>Chapter 2</title>
<qa_pair><label>例1</label><question>Calculate \(x\) such that \(x^2-1=0\).<pic>tag5:box7</pic></question>
<answer>\(x=\pm1\).</answer><solution>SOLUTION_TEXT</solution></qa_pair>
<qa_pair><label>例2</label><question>Calculate \(x\) such that \(x^2-4=0\).<pic>tag5:box8</pic></question>
<answer>\(x=\pm2\).</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>Chapter 3</title>
<qa_pair><label>例1</label><question>Calculate \(x\) such that \(x^3-1=0\).<pic>tag6:box7</pic></question>
<answer>\(x=1\).</answer><solution>SOLUTION_TEXT</solution></qa_pair>
</chapter>
<chapter><title>Chapter 4</title>
<qa_pair><label>0</label><question></question><answer></answer><solution></solution></qa_pair>
</chapter>

Please now process the provided page_n image and output your result.
"""
        return PROMPT
      
      
@PROMPT_REGISTRY.register()
class QAExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self) -> str:
        PROMPT = f"""
        You are an expert in answer college-level questions. You are given a json file. Your task is to segment the content, insert images tags, and extract labels:
1. Every json item has an "id" field. Your main task is to output this field.
2. You need to segment the content into multiple `<qa_pair>`…`</qa_pair>` blocks, each containing a question and its corresponding answer with solution.
3. If the problem or answer/solution is not complete, omit them. An answer/solution should be considered complete as long as either the answer or solution exists.
4. You need to put the images id into proper positions. You could look at the caption or context to decide where to put the image tags.
5. You will also need to extract the chapter title and each problem's label/number from the text.
6. You only need to output "id" field for **chapter titles, questions and solutions**. DO NOT OUTPUT ORIGINAL TEXT. Use ',' to separate different ids.
7. However, use original labels/numbers for labels, and use original numbers for answers. DO NOT output "id" field for labels and answers. You will need to extract them from the text.
"""
        PROMPT +=f"""
Strict extraction rules:
** About questions and answers/solutions **
- Preserve each problem’s original label/number, such as "例1", "Example 3", "习题1", "11". Do not include the period after the number. Use Arabic numerals only. For example, if the label is "例一", convert it to "例1". If the label is "IV", convert it to "4". 
- If the full label is "三、16", keep only "16". If the full label is "5.4", keep only "4".
- If there are multiple sub-questions (such as "(1)", "(a)") under one main question, always put them together in the same `<qa_pair>`…`</qa_pair>` block.
- If a question and its answer/solution are contiguous, wrap them together as a single `<qa_pair>`…`</qa_pair>` block, e.g.:
  `<qa_pair><label>1</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>`
- If a question and its answer/solution are NOT contiguous (e.g. only question; only answer and/or solution; all questions at the front and all answers/solutions at the back), wrap each question or answer/solution in a `<qa_pair>`…`</qa_pair>` block with the missing part left empty. For example, if only questions appear:
  `<qa_pair><label>1</label><question>…</question><answer></answer><solution></solution></qa_pair>`
- In total, there are 7 possibilities: only question, only answer, only solution, question with answer, question with solution, answer with solution, full question and answer and solution. 
- If multiple qa pairs appear, wrap each qa pair in its own `<qa_pair>`…`</qa_pair>` block.
- If you do not see the full solution, only extract the short answer and leave the solution empty. YOU MUST KEEP SHORT ANSWERS !!!
** About chapter/section titles **
- Always enclose qa pairs in a `<chapter>`…`</chapter>` block, where <title>MAIN_TITLE_ID</title> is the id of the chapter title or section title.
- Normally, chapter/section titles appear before the questions/answers in an independent json item.
- There could be multiple `<chapter>`…`</chapter>` blocks if multiple chapters/sections exist. 
- **Any title followed by a question/answer whose label/number is not 1, or title with a score such as "一、选择题（每题1分，共10分）", should NOT be extracted.**
- Do not use nested titles.
- Leave the title blank if there is no chapter title. 
** About figures/diagrams **
- Whenever the question or answer/solution refers to a figure or diagram, record its "id" in question/answer/solution just like other text content.
- You MUST include all images referenced in the question/answer/solution.


If no qualifying content is found, output:
<empty></empty>

Output format (all tags run together, no extra whitespace or newlines except between entries):
<chapter><title>MAIN_TITLE_ID</title>
<qa_pair><label>LABEL(EXTRACTED FROM TEXT)</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution>SOLUTION_IDS</solution></qa_pair>
<qa_pair><label>LABEL(EXTRACTED FROM TEXT)</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>MAIN_TITLE_ID</title>
<qa_pair><label>LABEL(EXTRACTED FROM TEXT)</label><question>QUESTION_IDS</question>
<answer>ANSWER(EXTRACTED FROM SOLUTION)</answer><solution>SOLUTION_IDS</solution></qa_pair>
</chapter>


Example:
<chapter><title>7</title>
<qa_pair><label>1</label><question>2,3</question>
<answer>Yes</answer><solution>5,6,7</solution></qa_pair>
<qa_pair><label>2</label><question>8,9,10</question>
<answer>3.14</answer><solution></solution></qa_pair>
</chapter>
<chapter><title>12</title>
<qa_pair><label>1</label><question></question>
<answer>2^6</answer><solution>16</solution></qa_pair>
</chapter>

Please now process the provided json and output your result.
"""
        return PROMPT