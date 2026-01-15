"""
Table Chunk Splitter Operator
Splits documents by table boundaries to create coherent chunks for processing.
"""
import re
import pandas as pd
from typing import Any, Dict, List, Optional

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class TableChunkSplitter(OperatorABC):
    """
    Split documents by table boundaries.
    Each chunk contains one or more tables with surrounding context.
    """

    def __init__(
        self,
        max_chunk_size: int = 20000,
        min_chunk_size: int = 500,
        context_before: int = 500,
        context_after: int = 500,
    ):
        """
        Initialize the TableChunkSplitter.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            context_before: Characters to include before table
            context_after: Characters to include after table
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.context_before = context_before
        self.context_after = context_after
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        if lang == "zh":
            return (
                "TableChunkSplitter 按表格边界切分文档",
                "每个chunk包含完整的表格及其上下文",
            )
        else:
            return (
                "TableChunkSplitter splits documents by table boundaries",
                "Each chunk contains complete tables with context",
            )

    def _find_table_positions(self, text: str) -> List[Dict[str, int]]:
        """Find start and end positions of all tables in text."""
        tables = []
        
        # Find HTML tables
        html_pattern = r'<table[^>]*>.*?</table>'
        for match in re.finditer(html_pattern, text, re.IGNORECASE | re.DOTALL):
            tables.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'html'
            })
        
        # Find Markdown tables if no HTML tables
        if not tables:
            md_pattern = r'(\|[^\n]+\|\n)+(\|[-:| ]+\|\n)?(\|[^\n]+\|\n?)+'
            for match in re.finditer(md_pattern, text):
                tables.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'markdown'
                })
        
        return sorted(tables, key=lambda x: x['start'])

    def _split_by_tables(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks based on table positions."""
        tables = self._find_table_positions(text)
        
        if not tables:
            # No tables found, return whole text as single chunk
            return [{'id': 0, 'text': text, 'has_table': False}]
        
        chunks = []
        last_end = 0
        
        for i, table in enumerate(tables):
            table_start = table['start']
            table_end = table['end']
            
            # Calculate context boundaries
            context_start = max(last_end, table_start - self.context_before)
            context_end = min(len(text), table_end + self.context_after)
            
            # Check if we should merge with previous chunk
            if chunks and (context_start - last_end) < self.min_chunk_size:
                # Merge with previous chunk
                chunks[-1]['text'] += text[last_end:context_end]
                chunks[-1]['has_table'] = True
            else:
                # Add non-table content before this table as separate chunk if large enough
                if context_start > last_end and (context_start - last_end) > self.min_chunk_size:
                    before_text = text[last_end:context_start]
                    chunks.append({
                        'id': len(chunks),
                        'text': before_text,
                        'has_table': False
                    })
                
                # Add table chunk
                chunk_text = text[context_start:context_end]
                chunks.append({
                    'id': len(chunks),
                    'text': chunk_text,  #原始文档文本
                    'has_table': True
                })
            
            last_end = context_end
        
        # Add remaining text after last table
        if last_end < len(text):
            remaining = text[last_end:]
            if len(remaining) > self.min_chunk_size:
                chunks.append({
                    'id': len(chunks),
                    'text': remaining,
                    'has_table': False
                })
            elif chunks:
                chunks[-1]['text'] += remaining
        
        # Split any chunks that exceed max size
        final_chunks = []
        for chunk in chunks:
            if len(chunk['text']) > self.max_chunk_size:
                # Split large chunks
                text_parts = self._split_large_chunk(chunk['text'])
                for part in text_parts:
                    final_chunks.append({
                        'id': len(final_chunks),
                        'text': part,
                        'has_table': chunk['has_table']
                    })
            else:
                chunk['id'] = len(final_chunks)
                final_chunks.append(chunk)
        
        return final_chunks

    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a large chunk into smaller pieces."""
        parts = []
        current_pos = 0
        
        while current_pos < len(text):
            end_pos = min(current_pos + self.max_chunk_size, len(text))
            
            # Try to find a good break point
            if end_pos < len(text):
                # Look for paragraph break
                break_point = text.rfind('\n\n', current_pos, end_pos)
                if break_point > current_pos + self.min_chunk_size:
                    end_pos = break_point + 2
                else:
                    # Look for sentence break
                    break_point = text.rfind('. ', current_pos, end_pos)
                    if break_point > current_pos + self.min_chunk_size:
                        end_pos = break_point + 2
            
            parts.append(text[current_pos:end_pos])
            current_pos = end_pos
        
        return parts

    def process(self, text: str) -> List[Dict[str, Any]]:
        """Process a single text and return chunks."""
        if not text or not isinstance(text, str):
            return []
        
        chunks = self._split_by_tables(text)
        self.logger.info(f"Split document into {len(chunks)} chunks")
        return chunks

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = 'text',
        output_key: str = 'chunks',
    ):
        """
        Run the table chunk splitter.
        
        Args:
            storage: DataFlowStorage instance
            input_key: Key for input text
            output_key: Key for output chunks
        """
        import os
        
        dataframe = storage.read("dataframe")
        
        # Handle text_path if text column doesn't exist
        if input_key not in dataframe.columns:
            if 'text_path' in dataframe.columns:
                self.logger.info("Reading text from text_path files...")
                texts = []
                for _, row in dataframe.iterrows():
                    text_path = row.get("text_path", "")
                    if text_path and os.path.exists(text_path):
                        with open(text_path, "r", encoding="utf-8") as f:
                            texts.append(f.read())
                    else:
                        texts.append("")
                dataframe[input_key] = texts
            else:
                raise ValueError(f"Column '{input_key}' not found in dataframe")
        
        all_chunks = []
        for idx, row in dataframe.iterrows():
            text = row.get(input_key, "")
            chunks = self.process(text)
            all_chunks.append(chunks)
        
        dataframe[output_key] = all_chunks
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key]
