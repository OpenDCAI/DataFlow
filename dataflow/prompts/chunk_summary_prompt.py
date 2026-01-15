"""
Prompt templates for chunk summarization in Map-Reduce QA pipeline
"""
import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC


@PROMPT_REGISTRY.register()
class ChunkSummaryPrompt(PromptABC):
    """
    Prompt for extracting structured summaries from document chunks.
    Used in the Map phase of Map-Reduce QA generation.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a document analysis expert. Your task is to extract structured information from document chunks.
                
                For each chunk, you must identify and extract:
                1. Tables present (name/description, column headers, data types)
                2. Key numerical values mentioned in text
                3. Conclusions or claims about the data
                4. Relationships between data points
                
                Output ONLY valid JSON with this structure:
                {
                    "chunk_id": <provided id>,
                    "tables": [
                        {
                            "description": "Brief description of table content",
                            "columns": ["col1", "col2", ...],
                            "data_types": ["numeric", "text", ...],
                            "row_count": <approximate rows>
                        }
                    ],
                    "key_values": [
                        {"name": "value description", "value": "123", "unit": "kg"}
                    ],
                    "claims": [
                        "Statement about data, e.g., 'Average temperature was 25°C'"
                    ],
                    "data_relationships": [
                        "Description of how data points relate to each other"
                    ]
                }
                """)
        else:
            return textwrap.dedent("""\
                您是文档分析专家。您的任务是从文档片段中提取结构化信息。
                
                对于每个片段，您必须识别和提取：
                1. 存在的表格（名称/描述、列标题、数据类型）
                2. 文本中提到的关键数值
                3. 关于数据的结论或声明
                4. 数据点之间的关系
                
                仅输出以下结构的有效JSON：
                {
                    "chunk_id": <提供的id>,
                    "tables": [
                        {
                            "description": "表格内容的简要描述",
                            "columns": ["列1", "列2", ...],
                            "data_types": ["数值", "文本", ...],
                            "row_count": <大约行数>
                        }
                    ],
                    "key_values": [
                        {"name": "值的描述", "value": "123", "unit": "千克"}
                    ],
                    "claims": [
                        "关于数据的陈述，例如'平均温度为25°C'"
                    ],
                    "data_relationships": [
                        "数据点之间关系的描述"
                    ]
                }
                """)

    def build_prompt(self, chunk_id: int, chunk_text: str) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Analyze the following document chunk (ID: {chunk_id}) and extract structured information.
                
                Document Chunk:
                {chunk_text}
                
                Extract all tables, key values, claims, and data relationships. Output valid JSON only.
                """)
        else:
            return textwrap.dedent(f"""\
                分析以下文档片段（ID: {chunk_id}）并提取结构化信息。
                
                文档片段：
                {chunk_text}
                
                提取所有表格、关键值、声明和数据关系。仅输出有效的JSON。
                """)


@PROMPT_REGISTRY.register()
class QAPlannerPrompt(PromptABC):
    """
    Prompt for planning which chunks to use for QA generation.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a QA planning expert. Given summaries of document chunks, you need to:
                1. Identify which chunks contain related data that can form verification questions
                2. Plan cross-chunk questions that reference data from multiple tables
                3. Prioritize questions that verify numerical claims with table data
                
                Output ONLY valid JSON array with this structure:
                [
                    {
                        "question_type": "verification|calculation|comparison",
                        "description": "Brief description of the question to generate",
                        "required_chunk_ids": [1, 3, 5],
                        "table_references": ["Table from chunk 1", "Table from chunk 3"],
                        "claim_to_verify": "The claim from text to verify against tables"
                    }
                ]
                """)
        else:
            return textwrap.dedent("""\
                您是QA规划专家。根据文档片段的摘要，您需要：
                1. 识别哪些片段包含可以形成验证问题的相关数据
                2. 规划引用多个表格数据的跨片段问题
                3. 优先考虑用表格数据验证数值声明的问题
                
                仅输出以下结构的有效JSON数组：
                [
                    {
                        "question_type": "验证|计算|比较",
                        "description": "要生成的问题的简要描述",
                        "required_chunk_ids": [1, 3, 5],
                        "table_references": ["片段1的表格", "片段3的表格"],
                        "claim_to_verify": "要用表格验证的文本声明"
                    }
                ]
                """)

    def build_prompt(self, summaries: str, max_questions: int = 10) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Based on the following chunk summaries, plan verification questions (UP TO {max_questions} maximum).
                
                IMPORTANT: Generate as many meaningful questions as the data supports, but no more than {max_questions}.
                - If the content only supports 3 good questions, generate 3.
                - If the content is rich with data, generate more (up to {max_questions}).
                - Focus on quality over quantity. Only generate questions that can be properly verified.
                
                Focus on questions that:
                - Can be answered by examining table data
                - Verify numerical claims made in the text
                - Compare data across different tables
                
                Chunk Summaries:
                {summaries}
                
                Output a JSON array of question plans. Generate only as many as the content meaningfully supports.
                """)
        else:
            return textwrap.dedent(f"""\
                根据以下片段摘要，规划验证问题（最多{max_questions}个）。
                
                重要：根据数据支持程度生成尽可能多的有意义问题，但不超过{max_questions}个。
                - 如果内容只能支持3个好问题，就生成3个。
                - 如果内容数据丰富，可以生成更多（最多{max_questions}个）。
                - 注重质量而非数量。只生成可以正确验证的问题。
                
                关注以下问题：
                - 可以通过检查表格数据来回答
                - 验证文本中的数值声明
                - 比较不同表格的数据
                
                片段摘要：
                {summaries}
                
                输出问题计划的JSON数组。只生成内容能够有意义支持的数量。
                """)


@PROMPT_REGISTRY.register()
class TargetedQAPrompt(PromptABC):
    """
    Prompt for generating QA from specific chunks based on QA plan.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.system_text = self.build_system_prompt()
        
    def build_system_prompt(self) -> str:
        if self.lang == "en":
            return textwrap.dedent("""\
                You are a QA generation expert. Generate a verification question based on:
                1. The question plan provided
                2. The relevant document chunks
                
                The question should:
                - Be answerable from the table data
                - Verify the claim mentioned in the plan
                - Include step-by-step reasoning
                
                CRITICAL: The answer MUST be a pure numerical value only!
                - Examples of valid answers: "49%", "2.5", "100", "3.14", "0.05%", "1.2 mg/L"
                - NO text explanations in the answer field
                - If the answer is a percentage, include the % symbol
                - If the answer has a unit, include the unit (e.g., "2.5 kg", "100 m")
                
                Output ONLY valid JSON:
                {
                    "question": "The question text",
                    "reasoning": "Step-by-step calculation or reasoning",
                    "answer": "NUMERICAL VALUE ONLY (e.g., 49%, 2.5, 100 mg/L)",
                    "table_reference": "Which table data was used",
                    "text_reference": "The claim being verified"
                }
                """)
        else:
            return textwrap.dedent("""\
                您是QA生成专家。根据以下内容生成验证问题：
                1. 提供的问题计划
                2. 相关的文档片段
                
                问题应该：
                - 可以从表格数据中回答
                - 验证计划中提到的声明
                - 包含逐步推理
                
                关键要求：答案必须是纯数值形式！
                - 有效答案示例："49%", "2.5", "100", "3.14", "0.05%", "1.2 mg/L"
                - 答案字段中不要有文字解释
                - 如果答案是百分比，包含%符号
                - 如果答案有单位，包含单位（如："2.5 kg", "100 m"）
                
                仅输出有效的JSON：
                {
                    "question": "问题文本",
                    "reasoning": "逐步计算或推理",
                    "answer": "仅数值（如：49%、2.5、100 mg/L）",
                    "table_reference": "使用了哪个表格数据",
                    "text_reference": "正在验证的声明"
                }
                """)

    def build_prompt(self, question_plan: str, chunks: str) -> str:
        if self.lang == "en":
            return textwrap.dedent(f"""\
                Generate a verification question based on the following plan and document chunks.
                
                Question Plan:
                {question_plan}
                
                Relevant Document Chunks:
                {chunks}
                
                Generate the question and answer with reasoning. Output valid JSON only.
                """)
        else:
            return textwrap.dedent(f"""\
                根据以下计划和文档片段生成验证问题。
                
                问题计划：
                {question_plan}
                
                相关文档片段：
                {chunks}
                
                生成带有推理的问题和答案。仅输出有效的JSON。
                """)
