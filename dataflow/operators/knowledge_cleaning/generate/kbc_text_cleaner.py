from dataflow.prompts.kbcleaning import KnowledgeCleanerPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from typing import Union

import re


_CLEANED_BLOCK_RE = re.compile(
    r"<cleaned_start>\s*(.*?)\s*<cleaned_end>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ANSWER_BLOCK_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)
_PROCESSING_STEPS_RE = re.compile(
    r"\s*Processing Steps:\s*"
    r"1\.\s*\[Tag Analysis\]\s*Classify markup tags\s*"
    r"2\.\s*\[Reference Extraction\]\s*Isolate images/tables\s*"
    r"3\.\s*\[Character Audit\]\s*Log special chars\s*"
    r"4\.\s*\[Structure Check\]\s*Validate hierarchy\s*"
    r"5\.\s*\[Final Output\]\s*Generate cleaned text.*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_PROCESSING_STEPS_WITHOUT_HEADER_RE = re.compile(
    r"\s*1\.\s*\[Tag Analysis\]\s*Classify markup tags\s*"
    r"2\.\s*\[Reference Extraction\]\s*Isolate images/tables\s*"
    r"3\.\s*\[Character Audit\]\s*Log special chars\s*"
    r"4\.\s*\[Structure Check\]\s*Validate hierarchy\s*"
    r"5\.\s*\[Final Output\]\s*Generate cleaned text.*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_ZH_PROCESSING_STEPS_RE = re.compile(
    r"\s*(?:处理步骤|处理流程|治理步骤)[:：]\s*"
    r"1[.、]\s*\[(?:标签分析|标记分析)\].*?"
    r"2[.、]\s*\[(?:引用提取|参考提取)\].*?"
    r"3[.、]\s*\[(?:字符审核|字符审计)\].*?"
    r"4[.、]\s*\[(?:结构检查|结构校验)\].*?"
    r"5[.、]\s*\[(?:最终输出|最终结果)\].*$",
    flags=re.DOTALL,
)
_ZH_PROCESSING_STEPS_WITHOUT_HEADER_RE = re.compile(
    r"\s*1[.、]\s*\[(?:标签分析|标记分析)\].*?"
    r"2[.、]\s*\[(?:引用提取|参考提取)\].*?"
    r"3[.、]\s*\[(?:字符审核|字符审计)\].*?"
    r"4[.、]\s*\[(?:结构检查|结构校验)\].*?"
    r"5[.、]\s*\[(?:最终输出|最终结果)\].*$",
    flags=re.DOTALL,
)


def extract_cleaned_text(text, post_process=None) -> str:
    """Extract model output while dropping prompt-instruction leakage."""
    text = "" if text is None else str(text)

    cleaned_match = _CLEANED_BLOCK_RE.search(text)
    if cleaned_match:
        text = cleaned_match.group(1)
    else:
        answer_match = _ANSWER_BLOCK_RE.search(text)
        if answer_match:
            text = answer_match.group(1)
        text = text.replace("<cleaned_start>", "").replace("<cleaned_end>", "")

    text = _PROCESSING_STEPS_RE.sub("", text)
    text = _PROCESSING_STEPS_WITHOUT_HEADER_RE.sub("", text)
    text = _ZH_PROCESSING_STEPS_RE.sub("", text)
    text = _ZH_PROCESSING_STEPS_WITHOUT_HEADER_RE.sub("", text)
    text = text.strip()

    if post_process:
        text = post_process(text)
    return text


@prompt_restrict(
    KnowledgeCleanerPrompt       
)

@OPERATOR_REGISTRY.register()
class KBCTextCleaner(OperatorABC):
    '''
        KnowledgeCleaner is a class that cleans knowledge for RAG to make them more accurate, reliable and readable.
    '''
    def __init__(self, llm_serving: LLMServingABC, lang="en", prompt_template : Union[KnowledgeCleanerPrompt, DIYPromptABC] = None):
        self.logger = get_logger()
        self.prompts = KnowledgeCleanerPrompt(lang=lang)    
        self.llm_serving = llm_serving
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt(lang=lang)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识清洗算子：对原始知识内容进行标准化处理，包括HTML标签清理、特殊字符规范化、"
                "链接处理和结构优化，提升RAG知识库的质量。主要功能：\n"
                "1. 移除冗余HTML标签但保留语义化标签\n"
                "2. 标准化引号/破折号等特殊字符\n"
                "3. 处理超链接同时保留文本\n"
                "4. 保持原始段落结构和代码缩进\n"
                "5. 确保事实性内容零修改\n"
                "\n输入格式示例：\n"
                "<div class=\"container\">\n"
                "  <h1>标题文本</h1>\n"
                "  <p>正文段落，包括特殊符号，例如“弯引号”、–破折号等</p>\n"
                "  <img src=\"example.jpg\" alt=\"示意图\">\n"
                "  <a href=\"...\">链接文本</a>\n"
                "  <pre><code>代码片段</code></pre>\n"
                "  ...\n"
                "</div>\n"
                "\n输出格式示例：\n"
                "标题文本\n\n"
                "正文段落，包括特殊符号，例如\"直引号\"、-破折号等\n\n"
                "[Image: 示例图 example.jpg]\n\n"
                "链接文本\n\n"
                "<code>代码片段</code>\n\n"
                "[结构保持，语义保留，敏感信息脱敏处理（如手机号、保密标记等）]"
            )
        elif lang == "en":
            return (
                "Knowledge Cleaning Operator: Standardizes raw HTML/text content for RAG quality improvement. Key functions:\n"
                "1. Removes redundant HTML tags while preserving semantic tags\n"
                "2. Normalizes special characters (e.g., curly quotes, dashes)\n"
                "3. Processes hyperlinks and retains their text\n"
                "4. Preserves paragraph structure and code indentation\n"
                "5. Ensures factual content remains unchanged\n"
                "\nExample Input Format:\n"
                "<div class=\"container\">\n"
                "  <h1>Title Text</h1>\n"
                "  <p>Paragraph with “curly quotes” and – dashes</p>\n"
                "  <img src=\"example.jpg\" alt=\"Diagram\">\n"
                "  <a href=\"...\">Link text</a>\n"
                "  <pre><code>Code block</code></pre>\n"
                "  ...\n"
                "</div>\n"
                "\nExample Output Format:\n"
                "Title Text\n\n"
                "Paragraph with \"straight quotes\" and - dashes\n\n"
                "[Image: Diagram example.jpg]\n\n"
                "Link text\n\n"
                "<code>Code block</code>\n\n"
                "[Structure retained, semantics preserved, sensitive info masked (e.g., phone numbers, confidential tags)]"
            )
        else:
            return "Knowledge cleaning operator for RAG content standardization. Set lang='zh' or 'en' for examples."



    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        raw_contents = dataframe[self.input_key].tolist()
        inputs = [self.prompt_template.build_prompt(raw_content) for raw_content in raw_contents]

        return inputs

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "raw_chunk", 
        output_key:str = "cleaned_chunk"
        ):
        '''
        Runs the knowledge cleaning process, reading from the input key and saving results to output key.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        cleaned = self.llm_serving.generate_from_input(formatted_prompts,"")

        # Save only the final cleaned text, even if the model leaks prompt steps.
        cleaned_extracted = [
            extract_cleaned_text(
                text,
                getattr(self.prompt_template, "_post_process", None),
            )
            for text in cleaned
        ]
        dataframe[self.output_key] = cleaned_extracted
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
