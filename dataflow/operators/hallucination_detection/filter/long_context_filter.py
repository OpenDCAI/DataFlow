"""
Long Context Filter Operator.

Filters samples based on token count to create long-context evaluation datasets.
Useful for benchmarking models with extended context windows (8K+, 12K+, 16K+, etc.).
"""

import pandas as pd
from typing import Optional, Union, List
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@OPERATOR_REGISTRY.register()
class LongContextFilterOperator(OperatorABC):
    """Filter samples by token count for long-context evaluation.
    
    This operator tokenizes text fields and filters samples based on
    minimum and maximum token counts. Useful for creating evaluation
    datasets that test models with extended context windows.
    
    Example:
        >>> from dataflow.operators.hallucination_detection import LongContextFilterOperator
        >>> from transformers import AutoTokenizer
        >>> 
        >>> tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        >>> filter_op = LongContextFilterOperator(
        ...     tokenizer=tokenizer,
        ...     min_tokens=8000,
        ...     max_tokens=24000,
        ... )
        >>> # Use in pipeline
    """
    
    def __init__(
        self,
        tokenizer: Optional["AutoTokenizer"] = None,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        min_tokens: int = 8000,
        max_tokens: int = 32000,
        text_fields: Optional[List[str]] = None,
        add_token_count: bool = True,
    ):
        """Initialize the LongContextFilterOperator.
        
        Args:
            tokenizer: Pre-loaded HuggingFace tokenizer. If None, loads from tokenizer_name.
            tokenizer_name: HuggingFace model name to load tokenizer from.
            min_tokens: Minimum token count (inclusive).
            max_tokens: Maximum token count (inclusive).
            text_fields: List of fields to concatenate for token counting.
                         Defaults to ["prompt", "answer"] or ["text"].
            add_token_count: If True, adds a 'num_tokens' column to output.
        """
        self.logger = get_logger()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for LongContextFilterOperator. "
                "Install with: pip install transformers"
            )
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.logger.info(f"Loading tokenizer from {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.text_fields = text_fields or ["prompt", "answer"]
        self.add_token_count = add_token_count
    
    @staticmethod
    def get_desc(lang: str = "en") -> str:
        """Returns a description of the operator's functionality."""
        if lang == "zh":
            return (
                "根据token数量过滤样本的算子，用于创建长上下文评估数据集。\n\n"
                "__init__参数：\n"
                "- tokenizer: HuggingFace tokenizer对象，用于token计数\n"
                "- tokenizer_name: tokenizer模型名称，默认'answerdotai/ModernBERT-base'\n"
                "- min_tokens: 最小token数（含），默认8000\n"
                "- max_tokens: 最大token数（含），默认32000\n"
                "- text_fields: 需要计算token的文本字段列表，默认['prompt', 'answer']\n"
                "- add_token_count: 是否添加num_tokens列，默认True\n\n"
                "run参数：\n"
                "- storage: DataFlow存储对象\n"
                "- input_key: 输入数据的键名\n"
                "- output_key: 输出数据的键名\n\n"
                "输出：过滤后的DataFrame，包含符合token范围的样本。"
            )
        else:
            return (
                "An operator that filters samples by token count for long-context evaluation datasets.\n\n"
                "__init__ Parameters:\n"
                "- tokenizer: HuggingFace tokenizer object for token counting\n"
                "- tokenizer_name: Tokenizer model name, default 'answerdotai/ModernBERT-base'\n"
                "- min_tokens: Minimum token count (inclusive), default 8000\n"
                "- max_tokens: Maximum token count (inclusive), default 32000\n"
                "- text_fields: List of text fields to count tokens from, default ['prompt', 'answer']\n"
                "- add_token_count: Whether to add num_tokens column, default True\n\n"
                "run Parameters:\n"
                "- storage: DataFlow storage object\n"
                "- input_key: Key for input data\n"
                "- output_key: Key for output data\n\n"
                "Output: Filtered DataFrame containing samples within the token range."
            )
    
    def _count_tokens(self, row: pd.Series) -> int:
        """Count tokens for a single row."""
        texts = []
        for field in self.text_fields:
            if field in row and pd.notna(row[field]):
                texts.append(str(row[field]))
        
        combined_text = " ".join(texts)
        tokens = self.tokenizer.encode(combined_text, add_special_tokens=True)
        return len(tokens)
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "dataframe",
        output_key: str = "filtered_dataframe",
    ) -> None:
        """Run the filter operation.
        
        Args:
            storage: DataFlow storage object containing the dataframe.
            input_key: Key for the input dataframe in storage.
            output_key: Key for the output filtered dataframe.
        """
        df = storage.get(input_key)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df)}")
        
        self.logger.info(f"Filtering {len(df)} samples by token count [{self.min_tokens}, {self.max_tokens}]")
        
        # Detect available text fields
        available_fields = [f for f in self.text_fields if f in df.columns]
        if not available_fields:
            # Fallback to 'text' if present
            if "text" in df.columns:
                available_fields = ["text"]
            else:
                raise ValueError(
                    f"None of the text_fields {self.text_fields} found in dataframe. "
                    f"Available columns: {list(df.columns)}"
                )
        
        self.text_fields = available_fields
        self.logger.info(f"Using text fields: {self.text_fields}")
        
        # Count tokens for each row
        from tqdm import tqdm
        tqdm.pandas(desc="Counting tokens")
        df["_token_count"] = df.progress_apply(self._count_tokens, axis=1)
        
        # Filter by token count
        mask = (df["_token_count"] >= self.min_tokens) & (df["_token_count"] <= self.max_tokens)
        filtered_df = df[mask].copy()
        
        # Rename or drop token count column
        if self.add_token_count:
            filtered_df = filtered_df.rename(columns={"_token_count": "num_tokens"})
        else:
            filtered_df = filtered_df.drop(columns=["_token_count"])
        
        # Log statistics
        self.logger.info(
            f"Filtered: {len(filtered_df)}/{len(df)} samples "
            f"({len(filtered_df)/len(df)*100:.1f}%) in token range [{self.min_tokens}, {self.max_tokens}]"
        )
        
        if len(filtered_df) > 0 and self.add_token_count:
            self.logger.info(
                f"Token stats: min={filtered_df['num_tokens'].min()}, "
                f"max={filtered_df['num_tokens'].max()}, "
                f"mean={filtered_df['num_tokens'].mean():.0f}"
            )
        
        storage.set(output_key, filtered_df)

