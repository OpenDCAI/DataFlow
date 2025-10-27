from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class VQAExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, subject: str = "math", interleaved=True) -> str:
        PROMPT = ""
        if interleaved:
          PROMPT = f"""
        You are an expert in {subject} competition. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All {subject} problems whose text begins on page_n and the answers with solutions to those problems.
2. If the problem or answer is not complete (because they continue onto page_n+1), omit them. If the problem is complete but the solution not, omit both the problem and the solution. DO NOT INCLUDE INCOMPLETE PROBLEMS OR ANSWERS. However, if only the solution is incomplete, you may still include the question and the short answer if they are complete.
3. Normally, a box at the beginning of a page with no problem number (such as "1.1", "例 1", "example 1", "解", "solution", "答案", "answer") is the continuation of the problem or solution from the previous page, even if it appears to be an independent paragraph. Omit them.
4. The chapter information as it appears on page_n.
"""
        else:
          PROMPT = f"""
        You are an expert in {subject} competition. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All {subject} problems whose text appears on page_n. In the provided page, there will be all questions or all answers with solutions, but not mixed.
2. If the problem or answer is not complete (because they continue onto page_n+1), omit them. DO NOT INCLUDE INCOMPLETE QUESTIONS OR ANSWERS. However, if only the solution is incomplete, you may still include the question and the short answer if they are complete.
3. Normally, a box at the beginning of a page with no problem number (such as "1.1", "例 1", "example 1", "解", "solution", "答案", "answer") is the continuation of the problem or solution from the previous page, even if it appears to be an independent paragraph. Omit them.
4. The chapter information as it appears on page_n.
"""
          PROMPT +="""
Strict extraction rules:
- If you think the page is not the main text page, such as a cover page, catalog page, header/footer only, etc., output `<empty></empty>`.
- Preserve each problem’s original label/number, such as "例1", "Example 3", "习题1", "11". Do not include the period after the number.
- If there are multiple sub-questions under one main question, always put them together in the same `<qa_pair>`…`</qa_pair>` block.
- If a question and its answer/solution are contiguous on page_n, wrap them together as a single `<qa_pair>`…`</qa_pair>` block, e.g.:
  `<qa_pair><label>例1</label><question>…</question><answer>…</answer><solution>…</solution></qa_pair>`
- If only questions or only answers with solutions appear on page_n, wrap each question or answer with solution in a `<qa_pair>`…`</qa_pair>` block with the missing part left empty. For example, if only questions appear:
  `<qa_pair><label>例1</label><question>…</question><answer></answer><solution></solution></qa_pair>`
- If multiple questions and solutions appear on page_n, wrap each question/solution pair in its own `<qa_pair>`…`</qa_pair>` block.
- Sometimes a short answer may appear before the full solution. If you do not see the full solution on page_n, only extract the short answer and leave the solution empty.
- Enclose the output in a `<chapter>`…`</chapter>` block, where <title>MAIN_TITLE</title> is the chapter title or section title appearing on page_n.
- There could be multiple `<chapter>`…`</chapter>` blocks if multiple chapters/sections appear on page_n.
- Extract chapter titles only (e.g., "2025 GDFZ 入学数学真卷(一)").  **Do not keep subtitles such as "三、解答题（15分）". Any titles appearing with scores are not chapter titles, do NOT extract them.**
- If you encounter a title with no problems or answers on the page, still extract the title within the `<chapter>`…`</chapter>` block, with an empty qa_pair block `<qa_pair><label></label><question></question><answer></answer><solution></solution></qa_pair>`.
- Do not use nested titles.
- Leave the title blank if there is no chapter title of the questions. 
- Sometimes the chapter title may not appear at the beginning of the page. You should leave the title of all the qa pairs before the chapter title **blank**, but the title of the qa pairs after the chapter title should use this chapter title.
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
<answer>ANSWER_TEXT<pic>…</pic>…</answer><solution>SOLUTION_TEXT</solution></qa_pair>
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
<answer>\(x=\pm2\).</answer><solution>SOLUTION_TEXT</solution></qa_pair>
</chapter>
<chapter><title>Chapter 3</title>
<qa_pair><label>例1</label><question>Calculate \(x\) such that \(x^3-1=0\).<pic>tag6:box7</pic></question>
<answer>\(x=1\).</answer><solution>SOLUTION_TEXT</solution></qa_pair>
</chapter>

Please now process the provided page_n image and output your result.
"""
        return PROMPT