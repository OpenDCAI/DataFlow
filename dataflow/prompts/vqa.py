from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC

@PROMPT_REGISTRY.register()
class VQAExtractPrompt(PromptABC):
    def __init__(self):
        pass

    def build_prompt(self, subject: str = "math") -> str:
        PROMPT = f"""
        You are an expert in {subject} competition. You are given an image—page_n—annotated with detected bounding boxes and corresponding labels. Your task is to extract from page_n only:
1. All {subject} problems whose text begins on page_n and the answers to those problems.
2. If the problem or answer is not complete (because they continue onto page_n+1), omit them. If the problem is complete but the answer not, omit both the problem and the answer. DO NOT INCLUDE INCOMPLETE PROBLEMS OR ANSWERS.
3. Normally, a box at the beginning of a page with no title (such as "1.1", "例 1", "example 1", "解", "solution", "答案", "answer") is the continuation of the problem or answer from the previous page, even if it appears to be an independent paragraph. Omit them.
4. The chapter information (main titles and subtitles) as it appears on page_n.

Strict extraction rules:
- If you think the page is not the main text page, such as a cover page, catalog page, header/footer only, etc., output `<empty></empty>`.
- Preserve each problem’s original label/number. If an answer box appears directly below its question, infer that it shares the same label.
- If a question and its answer/proof are contiguous on page_n, wrap them together as a single `<qa_pair>`…`</qa_pair>` block, e.g.:
  `<qa_pair><label>例1</label><question>…</question><answer>…</answer></qa_pair>`
- For problem and answer text, output exactly what appears (no translation). Render all mathematical expressions in LaTeX.
- Whenever the question or answer refers to a figure or diagram, record it with `<pic>tagA:boxB</pic>`, such as `<pic>tag5:box7</pic>`. tagA:boxB is labeled (in exactly the same format) in the image beside the figure or diagram in RED color. Be careful that the original caption of the book may also exist, but usually in format A.B (normally in black color). Do NOT use the original caption in the book!!! Additionally, the figure/diagram may be surrounded by multiple labels (some from other boxes), be careful to pick the correct one. The correct one will be at the upper right of the figure/diagram. If you are not sure, you are free to put multiple labels, e.g. `<pic>tag5:box7</pic> <pic>tag5:box8</pic>`. NEVER leave it blank or make up a label!
- You should always put the `<pic>...</pic>` tag at the exact position where the figure/diagram is referenced in the text. If there are multiple references, put multiple tags at the correct positions.
- Extract all headings that represent structural information in the main body of the text (e.g., chapter titles, section titles such as “习题1.a”, lecture headings like “第一讲 相似”). Treat composite titles as a single unit (so “第一讲 相似” is one title, not two). Do not extract running headers or footers.

If no qualifying content is found, output:
<empty></empty>

Output format (all tags run together, no extra whitespace or newlines except between entries):
<question><label>…</label>QUESTION_TEXT<pic>…</pic>…</question>
<answer><label>…</label>ANSWER_TEXT<pic>…</pic>…</answer>
<title>MAIN_TITLE_OR_SUBTITLE</title>
[repeat as needed or `<empty></empty>`]

Example (for page_1 & page_2):
<question><label>例1</label>Calculate \(x\) such that \(x^2-1=0\).<pic>tag5:box7</pic></question>
<answer><label>例1</label>\(x=\pm1\).</answer>
<title>Chapter 2</title>
<title>Section 2.1</title>

Please now process the provided page_n image and output your result.
"""
        return PROMPT