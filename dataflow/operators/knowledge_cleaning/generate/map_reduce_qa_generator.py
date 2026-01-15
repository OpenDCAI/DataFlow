"""
Map-Reduce QA Generator Operator
Generates Q/A pairs using Map-Reduce approach for long documents.
"""
import json
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.chunk_summary_prompt import ChunkSummaryPrompt, QAPlannerPrompt, TargetedQAPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC


@OPERATOR_REGISTRY.register()
class MapReduceQAGenerator(OperatorABC):
    """
    Generate Q/A pairs using Map-Reduce approach.
    
    Map phase: Generate summaries for each chunk
    Reduce phase: Plan and generate cross-chunk Q/A pairs
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        max_qa: int = 20,
        max_relevant_chunks: int = 3,
    ):
        """
        Initialize the MapReduceQAGenerator.
        
        Args:
            llm_serving: LLM serving instance
            lang: Language for prompts
            max_qa: Maximum number of Q/A pairs to generate (LLM decides actual count)
            max_relevant_chunks: Max chunks to include per question
        """
        self.llm_serving = llm_serving
        self.lang = lang
        self.max_qa = max_qa
        self.max_relevant_chunks = max_relevant_chunks
        self.logger = get_logger()
        
        self.summary_prompt = ChunkSummaryPrompt(lang=lang)
        self.planner_prompt = QAPlannerPrompt(lang=lang)
        self.qa_prompt = TargetedQAPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        if lang == "zh":
            return (
                "MapReduceQAGenerator 使用Map-Reduce方式处理长文档",
                "先生成每个chunk的摘要，再规划并生成跨chunk的Q/A对",
            )
        else:
            return (
                "MapReduceQAGenerator processes long documents using Map-Reduce",
                "First generates chunk summaries, then plans and generates cross-chunk Q/A pairs",
            )

    def _generate_chunk_summaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map phase: Generate summaries for each chunk."""
        self.logger.info(f"Generating summaries for {len(chunks)} chunks...")
        
        # Filter chunks with tables for priority processing
        table_chunks = [c for c in chunks if c.get('has_table', False)]
        non_table_chunks = [c for c in chunks if not c.get('has_table', False)]
        
        summaries = []
        
        # Process table chunks
        if table_chunks:
            user_inputs = []
            for chunk in table_chunks:
                prompt = self.summary_prompt.build_prompt(chunk['id'], chunk['text'])
                user_inputs.append(prompt)
            
            sys_prompt = self.summary_prompt.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=sys_prompt
            )
            
            for chunk, response in zip(table_chunks, responses):
                summary = self._parse_summary_response(response, chunk['id'])
                summaries.append(summary)
        
        # Process non-table chunks - also generate summaries via LLM
        # This is important because table-related answers may appear in text chunks
        if non_table_chunks:
            self.logger.info(f"Generating summaries for {len(non_table_chunks)} non-table chunks...")
            user_inputs = []
            for chunk in non_table_chunks:
                prompt = self.summary_prompt.build_prompt(chunk['id'], chunk['text'])
                user_inputs.append(prompt)
            
            sys_prompt = self.summary_prompt.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=sys_prompt
            )
            
            for chunk, response in zip(non_table_chunks, responses):
                summary = self._parse_summary_response(response, chunk['id'])
                summary['has_table'] = False  # Mark as non-table chunk
                summaries.append(summary)
        
        # Sort by chunk_id
        summaries.sort(key=lambda x: x.get('chunk_id', 0))
        return summaries

    def _parse_summary_response(self, response: str, chunk_id: int) -> Dict[str, Any]:
        """Parse LLM response for chunk summary."""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed['chunk_id'] = chunk_id
                parsed['has_table'] = True
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    parsed['chunk_id'] = chunk_id
                    parsed['has_table'] = True
                    return parsed
        except:
            pass
        
        # Return minimal summary if parsing fails
        return {
            'chunk_id': chunk_id,
            'tables': [],
            'key_values': [],
            'claims': [],
            'data_relationships': [],
            'has_table': True,
            'parse_error': True
        }

    def _plan_qa_questions(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan which questions to generate based on summaries."""
        self.logger.info("Planning Q/A questions based on summaries...")
        
        # Filter summaries with tables
        table_summaries = [s for s in summaries if s.get('has_table', False)]
        
        if not table_summaries:
            self.logger.warning("No summaries with tables found")
            return []
        
        # Create condensed summary text for planning
        summary_text = json.dumps(table_summaries, ensure_ascii=False, indent=2)
        
        # Check if summary is too long
        if len(summary_text) > 50000:
            # Truncate each summary
            truncated = []
            for s in table_summaries:
                truncated.append({
                    'chunk_id': s.get('chunk_id'),
                    'tables': s.get('tables', [])[:2],
                    'claims': s.get('claims', [])[:3],
                    'key_values': s.get('key_values', [])[:5]
                })
            summary_text = json.dumps(truncated, ensure_ascii=False, indent=2)
        
        prompt = self.planner_prompt.build_prompt(summary_text, self.max_qa)
        sys_prompt = self.planner_prompt.build_system_prompt()
        
        responses = self.llm_serving.generate_from_input(
            user_inputs=[prompt],
            system_prompt=sys_prompt
        )
        
        return self._parse_qa_plan(responses[0]) if responses else []

    def _parse_qa_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse QA plan from LLM response."""
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return parsed
        except:
            pass
        
        return []

    def _generate_qa_from_plan(
        self,
        qa_plans: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reduce phase: Generate Q/A pairs from plans."""
        self.logger.info(f"Generating {len(qa_plans)} Q/A pairs...")
        
        if not qa_plans:
            return []
        
        # Create chunk lookup
        chunk_dict = {c['id']: c for c in chunks}
        
        qa_pairs = []
        user_inputs = []
        valid_plans = []
        
        for plan in qa_plans:
            chunk_ids = plan.get('required_chunk_ids', [])[:self.max_relevant_chunks]
            
            if not chunk_ids:
                continue
            
            # Get relevant chunks
            relevant_chunks = []
            for cid in chunk_ids:
                if cid in chunk_dict:
                    relevant_chunks.append(chunk_dict[cid]['text'])
            
            if not relevant_chunks:
                continue
            
            chunks_text = "\n\n---\n\n".join(relevant_chunks)
            
            # Check token limit (rough estimate: 4 chars per token)
            if len(chunks_text) > 80000:  # ~20k tokens
                chunks_text = chunks_text[:80000] + "...[truncated]"
            
            plan_text = json.dumps(plan, ensure_ascii=False)
            prompt = self.qa_prompt.build_prompt(plan_text, chunks_text)
            user_inputs.append(prompt)
            valid_plans.append(plan)
        
        if not user_inputs:
            return []
        
        sys_prompt = self.qa_prompt.build_system_prompt()
        responses = self.llm_serving.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=sys_prompt
        )
        
        for response in responses:
            qa = self._parse_qa_response(response)
            if qa:
                qa_pairs.append(qa)
        
        return qa_pairs

    def _parse_qa_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse Q/A from LLM response."""
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and 'question' in parsed and 'answer' in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict) and 'question' in parsed and 'answer' in parsed:
                    return parsed
        except:
            pass
        
        return None

    def _extract_tables_to_csv(
        self, 
        chunks: List[Dict[str, Any]], 
        output_dir: str,
        base_name: str = "doc"
    ) -> List[str]:
        """
        Extract tables from chunks and save as CSV files.
        
        Args:
            chunks: List of document chunks
            output_dir: Directory to save CSV files
            base_name: Base name for CSV files
            
        Returns:
            List of paths to saved CSV files
        """
        import os
        from io import StringIO
        
        os.makedirs(output_dir, exist_ok=True)
        csv_paths = []
        
        # Extract HTML tables from each chunk
        html_table_pattern = r'<table[^>]*>(.*?)</table>'
        table_idx = 0
        
        for chunk in chunks:
            text = chunk.get('text', '')
            chunk_id = chunk.get('id', 0)
            
            html_tables = re.findall(html_table_pattern, text, re.IGNORECASE | re.DOTALL)
            
            for table_html in html_tables:
                try:
                    # Wrap with table tags for parsing
                    full_table = f"<table>{table_html}</table>"
                    
                    # Use pandas to parse HTML table
                    dfs = pd.read_html(StringIO(full_table))
                    if dfs:
                        table_idx += 1
                        csv_path = os.path.join(output_dir, f"{base_name}_chunk{chunk_id}_table{table_idx}.csv")
                        dfs[0].to_csv(csv_path, index=False, encoding='utf-8')
                        csv_paths.append(csv_path)
                        self.logger.info(f"Saved table {table_idx} from chunk {chunk_id} to {csv_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse HTML table in chunk {chunk_id}: {e}")
        
        self.logger.info(f"Extracted {len(csv_paths)} tables to {output_dir}")
        return csv_paths

    def process(self, chunks: List[Dict[str, Any]], csv_output_dir: str = None) -> Dict[str, Any]:
        """
        Process chunks using Map-Reduce approach.
        
        Args:
            chunks: List of document chunks
            csv_output_dir: Optional directory to save extracted tables as CSV
            
        Returns:
            Dictionary with qa_pairs, summaries, plans and csv_paths
        """
        if not chunks:
            return {'qa_pairs': [], 'summaries': [], 'plans': [], 'csv_paths': []}
        
        # Extract tables to CSV if output directory is specified
        csv_paths = []
        if csv_output_dir:
            csv_paths = self._extract_tables_to_csv(chunks, csv_output_dir)
        
        # Map phase
        summaries = self._generate_chunk_summaries(chunks)   #生成chunk的摘要
        
        # Planning phase
        qa_plans = self._plan_qa_questions(summaries)   #用摘要规划Q/A问题
        
        # Reduce phase
        qa_pairs = self._generate_qa_from_plan(qa_plans, chunks)   #用chunk生成Q/A对
        
        return {
            'qa_pairs': qa_pairs,
            'summaries': summaries,
            'plans': qa_plans,
            'csv_paths': csv_paths
        }

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = 'chunks',
        output_key: str = 'table_qa_pairs',
    ):
        """
        Run the Map-Reduce QA Generator.
        
        Args:
            storage: DataFlowStorage instance
            input_key: Key for input chunks
            output_key: Key for output QA pairs
        """
        dataframe = storage.read("dataframe")
        
        if input_key not in dataframe.columns:
            raise ValueError(f"Column '{input_key}' not found in dataframe")
        
        all_results = []
        for idx, row in dataframe.iterrows():
            chunks = row.get(input_key, [])
            if not chunks:
                all_results.append({'qa_pairs': [], 'summaries': [], 'plans': []})
                continue
            
            result = self.process(chunks)
            all_results.append(result)
            self.logger.info(f"Generated {len(result['qa_pairs'])} Q/A pairs for document {idx}")
        
        dataframe[output_key] = [r['qa_pairs'] for r in all_results]
        dataframe['chunk_summaries'] = [r['summaries'] for r in all_results]
        dataframe['qa_plans'] = [r['plans'] for r in all_results]
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key, 'chunk_summaries', 'qa_plans']
