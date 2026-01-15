"""
A collection of prompts for the PDF Table QA pipeline operator
"""
import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class PDFTableQAGeneratorPrompt(PromptABC):
    '''
    PDF表格QA生成器提示模板（严格JSON格式输出）
    根据表格数据和文字中的计算结论生成验证性Q/A
    '''
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        """构建表格QA生成提示"""
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a professional Table Data QA Specialist with strict protocols:

                █ Core Requirements
                1. Identify tables in the provided markdown content
                2. Find mentions of computed values (averages, sums, percentages, etc.) in the text
                3. Generate Q/A pairs that verify the relationship between:
                   - Table data (raw numbers)
                   - Textual conclusions/computed values mentioned in the document
                4. Questions should require reasoning over table data
                5. Answers must be verifiable by examining the table
                                   
                █ Output Specifications
                Output ONLY pure JSON array with this structure:
                [
                    {
                        "question": "Question about data from table",
                        "reasoning": "Step-by-step calculation or reasoning",
                        "answer": "Numerical answer or conclusion",
                        "table_reference": "Brief description of which table data was used",
                        "text_reference": "The statement in text being verified"
                    }
                ]

                █ Question Types to Generate
                1. Verification questions: "Is the average of X correct as stated?"
                2. Calculation questions: "What is the sum/average/max of column X?"
                3. Comparison questions: "Which item has the highest/lowest value?"
                4. Trend questions: "How does value X compare to value Y?"

                █ Rejection Criteria
                Reject if:
                - No tables found in content
                - No numerical data to verify
                - Any non-JSON content appears
                """)
        else:
            return textwrap.dedent("""\
                您是专业的表格数据QA生成专家，必须严格遵循以下标准：

                █ 核心要求
                1. 识别提供的Markdown内容中的表格
                2. 找出文本中提到的计算值（平均值、总和、百分比等）
                3. 生成验证以下关系的Q/A对：
                   - 表格数据（原始数字）
                   - 文档中提到的文字结论/计算值
                4. 问题应需要基于表格数据进行推理
                5. 答案必须可通过检查表格来验证
                                   
                █ 输出规范
                仅输出以下结构的纯JSON数组：
                [
                    {
                        "question": "关于表格数据的问题",
                        "reasoning": "逐步计算或推理过程",
                        "answer": "数值答案或结论",
                        "table_reference": "所使用的表格数据的简要描述",
                        "text_reference": "正在验证的文本陈述"
                    }
                ]

                █ 生成的问题类型
                1. 验证性问题："文中所述的X平均值是否正确？"
                2. 计算性问题："X列的总和/平均值/最大值是多少？"
                3. 比较性问题："哪个项目的值最高/最低？"
                4. 趋势性问题："X值与Y值相比如何？"

                █ 拒绝标准
                若出现以下情况则拒绝：
                - 内容中未找到表格
                - 没有可验证的数值数据
                - 出现任何非JSON内容
                """)

    def build_prompt(self, text: str) -> str:
        """生成表格QA的用户提示"""
        if self.lang == "en":
            user_prompt = textwrap.dedent(f"""\
            Analyze the following markdown content which contains tables and text.
            Generate Q/A pairs that verify the numerical claims in the text against the table data.

            Content:
            {text}

            Requirements:
            1. Find all tables in the content
            2. Identify any computed values mentioned in text (averages, sums, percentages, etc.)
            3. Generate Q/A pairs that test understanding of the table data
            4. Focus on questions where the answer can be computed from table data
            5. Output ONLY valid JSON array, no other text
            """)
        else:
            user_prompt = textwrap.dedent(f"""\
            分析以下包含表格和文字的Markdown内容。
            生成Q/A对以验证文本中的数值声明与表格数据的一致性。

            内容：
            {text}

            要求：
            1. 找出内容中的所有表格
            2. 识别文本中提到的任何计算值（平均值、总和、百分比等）
            3. 生成测试对表格数据理解的Q/A对
            4. 关注可从表格数据计算得出答案的问题
            5. 仅输出有效的JSON数组，不包含其他文本
            """)
        
        return user_prompt
