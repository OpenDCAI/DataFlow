"""
Span Annotation Operator.

Converts document-level hallucination labels to span-level annotations using
Natural Language Inference (NLI). Useful for converting datasets like HaluEval
to token-classification format.
"""

import pandas as pd
import re
from typing import Optional, List
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from tqdm import tqdm

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@OPERATOR_REGISTRY.register()
class SpanAnnotationOperator(OperatorABC):
    """Convert document-level labels to span-level using NLI.
    
    This operator takes answers with document-level hallucination labels
    and identifies which specific sentences are hallucinated using NLI.
    
    Example:
        >>> from dataflow.operators.hallucination_detection import SpanAnnotationOperator
        >>> 
        >>> annotator = SpanAnnotationOperator(
        ...     nli_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        ...     contradiction_threshold=0.7,
        ... )
    """
    
    def __init__(
        self,
        nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        contradiction_threshold: float = 0.7,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        """Initialize the SpanAnnotationOperator.
        
        Args:
            nli_model: HuggingFace model for NLI classification.
            contradiction_threshold: Threshold for labeling as contradiction.
            device: Device to run the model on ("cuda" or "cpu").
            batch_size: Batch size for NLI inference.
        """
        self.logger = get_logger()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for SpanAnnotationOperator. "
                "Install with: pip install transformers"
            )
        
        self.nli_model_name = nli_model
        self.contradiction_threshold = contradiction_threshold
        self.device = device
        self.batch_size = batch_size
        self.nli_pipeline = None  # Lazy loading
    
    def _load_nli_pipeline(self):
        """Lazy load the NLI pipeline."""
        if self.nli_pipeline is None:
            self.logger.info(f"Loading NLI model: {self.nli_model_name}")
            self.nli_pipeline = pipeline(
                "zero-shot-classification",
                model=self.nli_model_name,
                device=0 if self.device == "cuda" else -1,
            )
    
    @staticmethod
    def get_desc(lang: str = "en") -> str:
        """Returns a description of the operator's functionality."""
        if lang == "zh":
            return (
                "使用NLI将文档级幻觉标签转换为span级标注的算子。\n\n"
                "__init__参数：\n"
                "- nli_model: NLI模型名称，默认'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'\n"
                "- contradiction_threshold: 矛盾判定阈值，默认0.7\n"
                "- device: 运行设备'cuda'或'cpu'，默认'cuda'\n"
                "- batch_size: NLI推理批次大小，默认8\n\n"
                "run参数：\n"
                "- storage: DataFlow存储对象\n"
                "- input_key: 输入数据的键名\n"
                "- output_key: 输出数据的键名\n"
                "- input_context_field: 上下文字段名，默认'context'\n"
                "- input_answer_field: 答案字段名，默认'answer'\n"
                "- input_is_hallucinated_field: 幻觉标记字段名，默认'is_hallucinated'\n\n"
                "输出：DataFrame包含labels字段（含text、start、end、confidence）。"
            )
        else:
            return (
                "An operator that converts document-level hallucination labels to span-level using NLI.\n\n"
                "__init__ Parameters:\n"
                "- nli_model: NLI model name, default 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'\n"
                "- contradiction_threshold: Threshold for contradiction detection, default 0.7\n"
                "- device: 'cuda' or 'cpu', default 'cuda'\n"
                "- batch_size: Batch size for NLI inference, default 8\n\n"
                "run Parameters:\n"
                "- storage: DataFlow storage object\n"
                "- input_key: Key for input data\n"
                "- output_key: Key for output data\n"
                "- input_context_field: Column name for context, default 'context'\n"
                "- input_answer_field: Column name for answer, default 'answer'\n"
                "- input_is_hallucinated_field: Column for hallucination flag, default 'is_hallucinated'\n\n"
                "Output: DataFrame with labels field containing text, start, end, confidence."
            )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_sentence_position(self, text: str, sentence: str) -> tuple:
        """Find the start and end position of a sentence in text."""
        start = text.find(sentence)
        if start == -1:
            return None, None
        end = start + len(sentence)
        return start, end
    
    def _check_contradiction(self, premise: str, hypothesis: str) -> float:
        """Check if hypothesis contradicts premise using NLI."""
        self._load_nli_pipeline()
        
        try:
            result = self.nli_pipeline(
                hypothesis,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
                multi_label=False,
            )
            
            # Find contradiction score
            for label, score in zip(result["labels"], result["scores"]):
                if label == "contradiction":
                    return score
            return 0.0
        except Exception as e:
            self.logger.warning(f"NLI check failed: {e}")
            return 0.0
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "dataframe",
        output_key: str = "annotated_dataframe",
        input_context_field: str = "context",
        input_answer_field: str = "answer",
        input_is_hallucinated_field: str = "is_hallucinated",
    ) -> None:
        """Run the span annotation operation.
        
        Args:
            storage: DataFlow storage object.
            input_key: Key for the input dataframe.
            output_key: Key for the output dataframe.
            input_context_field: Column name for the reference context.
            input_answer_field: Column name for the answer.
            input_is_hallucinated_field: Column name indicating if sample is hallucinated.
        """
        df = storage.get(input_key)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df)}")
        
        # Validate required columns
        for col in [input_context_field, input_answer_field]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.logger.info(f"Annotating {len(df)} samples with span-level labels")
        
        results = []
        stats = {"total": 0, "annotated": 0, "spans_found": 0}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating spans"):
            result = row.to_dict()
            result["labels"] = []
            
            answer = row[input_answer_field]
            context = row[input_context_field]
            is_hallucinated = row.get(input_is_hallucinated_field, True)
            
            if is_hallucinated:
                # Split answer into sentences
                sentences = self._split_sentences(answer)
                
                for sentence in sentences:
                    if len(sentence) < 10:  # Skip very short sentences
                        continue
                    
                    # Check contradiction
                    score = self._check_contradiction(context, sentence)
                    
                    if score >= self.contradiction_threshold:
                        start, end = self._find_sentence_position(answer, sentence)
                        if start is not None:
                            result["labels"].append({
                                "text": sentence,
                                "start": start,
                                "end": end,
                                "label": "hallucinated",
                                "confidence": score,
                            })
                            stats["spans_found"] += 1
                
                if result["labels"]:
                    stats["annotated"] += 1
            
            stats["total"] += 1
            results.append(result)
        
        output_df = pd.DataFrame(results)
        
        # Log statistics
        self.logger.info(
            f"Annotation complete: {stats['annotated']}/{stats['total']} samples annotated, "
            f"{stats['spans_found']} total spans found"
        )
        
        storage.set(output_key, output_df)

