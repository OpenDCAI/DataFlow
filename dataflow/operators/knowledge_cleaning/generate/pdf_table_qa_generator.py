"""
PDF Table QA Generator Operator
Generates Q/A pairs from PDF tables and text descriptions
"""
import pandas as pd
import json
import re
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.pdf_table_qa import PDFTableQAGeneratorPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC


@prompt_restrict(PDFTableQAGeneratorPrompt)
@OPERATOR_REGISTRY.register()
class PDFTableQAGenerator(OperatorABC):
    """
    A processor for generating Q/A pairs from PDF tables and text.
    
    This operator identifies tables in markdown content (converted from PDF),
    finds numerical claims in the text, and generates verification Q/A pairs
    that test understanding of the table data.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[PDFTableQAGeneratorPrompt, DIYPromptABC] = None,
        max_qa: int = 20,
        min_text_length: int = 50,
        max_text_length: int = 100000,
    ):
        """
        Initialize the PDFTableQAGenerator.

        Args:
            llm_serving: LLM serving instance for generation
            seed: Random seed
            lang: Language for prompts ('en' or 'zh')
            prompt_template: Custom prompt template
            max_qa: Maximum number of Q/A pairs to generate per chunk (LLM decides actual count)
            min_text_length: Minimum text length to process
            max_text_length: Maximum text length to process
        """
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.max_qa = max_qa
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = PDFTableQAGeneratorPrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        """Returns a description of the processor's functionality."""
        if lang == "zh":
            return (
                "PDFTableQAGenerator 是表格QA生成处理器，支持从PDF转换的Markdown中识别表格数据并生成验证性问答对。",
                "处理流程包括：识别Markdown表格、提取文字中的数值描述、生成基于表格数据的Q/A对。",
                "输出格式如下：",
                "输入：\n"
                "text: <包含表格的Markdown内容>",
                "输出：\n"
                "{\n"
                "  \"qa_pairs\": [\n"
                "    {\n"
                "      \"question\": <关于表格数据的问题>,\n"
                "      \"reasoning\": <推理过程>,\n"
                "      \"answer\": <答案>,\n"
                "      \"table_reference\": <表格引用>,\n"
                "      \"text_reference\": <文本引用>\n"
                "    }\n"
                "  ]\n"
                "}"
            )
        else:
            return (
                "PDFTableQAGenerator is a processor for generating Q/A pairs from PDF-converted markdown with tables.",
                "It includes: table recognition in markdown, numerical claim extraction, and table-based Q/A generation.",
                "Expected output format:",
                "Input:\n"
                "text: <markdown content with tables>",
                "Output:\n"
                "{\n"
                "  \"qa_pairs\": [\n"
                "    {\n"
                "      \"question\": <question about table data>,\n"
                "      \"reasoning\": <reasoning steps>,\n"
                "      \"answer\": <answer>,\n"
                "      \"table_reference\": <table reference>,\n"
                "      \"text_reference\": <text reference>\n"
                "    }\n"
                "  ]\n"
                "}"
            )

    def _has_table(self, text: str) -> bool:
        """Check if markdown text contains a table (HTML or Markdown format)."""
        # HTML table pattern
        html_table_pattern = r'<table[^>]*>.*?</table>'
        if re.search(html_table_pattern, text, re.IGNORECASE | re.DOTALL):
            return True
        
        # Markdown table pattern: lines with | separators
        md_table_pattern = r'\|[^\n]+\|'
        if re.search(md_table_pattern, text):
            return True
        
        return False

    def _extract_tables_to_csv(self, text: str, output_dir: str, base_name: str) -> List[str]:
        """
        Extract tables from HTML/Markdown content and save as CSV files.
        
        Args:
            text: The markdown/HTML content containing tables
            output_dir: Directory to save CSV files
            base_name: Base name for CSV files
            
        Returns:
            List of paths to saved CSV files
        """
        import os
        from io import StringIO
        
        os.makedirs(output_dir, exist_ok=True)
        csv_paths = []
        
        # Extract HTML tables
        html_table_pattern = r'<table[^>]*>(.*?)</table>'
        html_tables = re.findall(html_table_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for idx, table_html in enumerate(html_tables):
            try:
                # Wrap with table tags for parsing
                full_table = f"<table>{table_html}</table>"
                
                # Use pandas to parse HTML table
                dfs = pd.read_html(StringIO(full_table))
                if dfs:
                    csv_path = os.path.join(output_dir, f"{base_name}_table_{idx + 1}.csv")
                    dfs[0].to_csv(csv_path, index=False, encoding='utf-8')
                    csv_paths.append(csv_path)
                    self.logger.info(f"Saved table {idx + 1} to {csv_path}")
            except Exception as e:
                self.logger.warning(f"Failed to parse HTML table {idx + 1}: {e}")
        
        # Extract Markdown tables if no HTML tables found
        if not csv_paths:
            md_table_pattern = r'(\|[^\n]+\|\n)+(\|[-:| ]+\|\n)?(\|[^\n]+\|\n?)+'
            md_tables = re.findall(md_table_pattern, text)
            
            for idx, table_match in enumerate(md_tables):
                try:
                    # Reconstruct markdown table
                    table_text = ''.join(table_match) if isinstance(table_match, tuple) else table_match
                    
                    # Parse markdown table manually
                    lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
                    if len(lines) >= 2:
                        # Parse header
                        header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
                        
                        # Skip separator line if present
                        start_idx = 1
                        if len(lines) > 1 and all(c in '-:|' for c in lines[1].replace(' ', '')):
                            start_idx = 2
                        
                        # Parse data rows
                        data = []
                        for line in lines[start_idx:]:
                            row = [cell.strip() for cell in line.split('|') if cell.strip()]
                            if row:
                                data.append(row)
                        
                        if header and data:
                            df = pd.DataFrame(data, columns=header[:len(data[0])] if len(header) >= len(data[0]) else header + [''] * (len(data[0]) - len(header)))
                            csv_path = os.path.join(output_dir, f"{base_name}_table_{idx + 1}.csv")
                            df.to_csv(csv_path, index=False, encoding='utf-8')
                            csv_paths.append(csv_path)
                            self.logger.info(f"Saved markdown table {idx + 1} to {csv_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse Markdown table {idx + 1}: {e}")
        
        return csv_paths

    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        if not isinstance(text, str):
            return ''
        
        text = text.strip()
        
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            self.logger.warning(f"Text length {len(text)} out of range [{self.min_text_length}, {self.max_text_length}]")
            return ''
        
        return text

    def _extract_qa_pairs(self, responses: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract Q/A pairs from LLM responses."""
        all_qa_pairs = []
        
        for response in responses:
            qa_pairs = []
            
            # Try to parse as JSON array directly
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            qa_pairs.append(item)
                    all_qa_pairs.append(qa_pairs)
                    continue
            except json.JSONDecodeError:
                pass
            
            # Try to find JSON array in response
            try:
                # Find content between [ and ]
                bracket_count = 0
                start_pos = -1
                
                for i, char in enumerate(response):
                    if char == '[':
                        if bracket_count == 0:
                            start_pos = i
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0 and start_pos != -1:
                            json_str = response[start_pos:i+1]
                            try:
                                parsed = json.loads(json_str)
                                if isinstance(parsed, list):
                                    for item in parsed:
                                        if isinstance(item, dict) and "question" in item and "answer" in item:
                                            qa_pairs.append(item)
                            except json.JSONDecodeError:
                                pass
                            start_pos = -1
                
            except Exception as e:
                self.logger.warning(f"Failed to parse QA response: {e}")
            
            all_qa_pairs.append(qa_pairs)
        
        return all_qa_pairs

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts in batch."""
        results = []
        user_inputs = []
        valid_indices = []
        
        for idx, text in enumerate(texts):
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                results.append({"qa_pairs": [], "has_table": False})
                continue
            
            has_table = self._has_table(processed_text)
            if not has_table:
                self.logger.info(f"No table found in text at index {idx}")
                results.append({"qa_pairs": [], "has_table": False})
                continue
            
            user_inputs.append(self.prompt_template.build_prompt(processed_text))
            valid_indices.append(idx)
            results.append(None)  # Placeholder
        
        if user_inputs:
            sys_prompt = self.prompt_template.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs, 
                system_prompt=sys_prompt
            )
            
            qa_pairs_list = self._extract_qa_pairs(responses)
            
            for i, idx in enumerate(valid_indices):
                qa_pairs = qa_pairs_list[i] if i < len(qa_pairs_list) else []
                results[idx] = {
                    "qa_pairs": qa_pairs[:self.max_qa] if len(qa_pairs) > self.max_qa else qa_pairs,
                    "has_table": True
                }
        
        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Validate input dataframe."""
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = 'text',
        input_text_path_key: str = 'text_path',
        output_key: str = 'table_qa_pairs',
        output_has_table_key: str = 'has_table',
        output_csv_paths_key: str = 'table_csv_paths',
        extract_csv: bool = True,
        csv_output_dir: str = './table_csv_output',
    ):
        """
        Run the PDF Table QA Generator.
        
        Args:
            storage: DataFlowStorage instance
            input_key: Key for input text content (if available)
            input_text_path_key: Key for input text file path (fallback if input_key not found)
            output_key: Key for output QA pairs
            output_has_table_key: Key indicating if table was found
            output_csv_paths_key: Key for output CSV file paths
            extract_csv: Whether to extract tables to CSV files
            csv_output_dir: Directory to save CSV files
        """
        import os
        
        self.input_key = input_key
        self.output_key = output_key
        
        dataframe = storage.read("dataframe")
        
        # If input_key column doesn't exist, try to read from text_path files
        if input_key not in dataframe.columns:
            if input_text_path_key not in dataframe.columns:
                raise ValueError(f"Neither '{input_key}' nor '{input_text_path_key}' found in dataframe columns")
            
            self.logger.info(f"Reading text content from '{input_text_path_key}' files...")
            texts = []
            for _, row in dataframe.iterrows():
                text_path = row.get(input_text_path_key, "")
                if text_path and os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                else:
                    texts.append("")
            dataframe[input_key] = texts
        
        texts = dataframe[self.input_key].tolist()
        self.logger.info(f"Processing {len(texts)} texts for table QA generation")
        
        outputs = self.process_batch(texts)
        
        dataframe[self.output_key] = [output['qa_pairs'] for output in outputs]
        dataframe[output_has_table_key] = [output['has_table'] for output in outputs]
        
        # Extract tables to CSV if enabled
        if extract_csv:
            all_csv_paths = []
            for idx, (text, output) in enumerate(zip(texts, outputs)):
                if output['has_table']:
                    csv_paths = self._extract_tables_to_csv(
                        text=text,
                        output_dir=csv_output_dir,
                        base_name=f"doc_{idx}"
                    )
                    all_csv_paths.append(csv_paths)
                else:
                    all_csv_paths.append([])
            dataframe[output_csv_paths_key] = all_csv_paths
            self.logger.info(f"Extracted tables saved to {csv_output_dir}")
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key, output_has_table_key, output_csv_paths_key] if extract_csv else [output_key, output_has_table_key]
