"""
Hallucination Injection Operator.

Injects RAGTruth-style hallucinations into LLM-generated answers for creating
evaluation datasets. Supports multiple hallucination types:
- Evident Conflict: Direct contradiction of source facts
- Evident Baseless Info: Fabricated facts not in source
- Subtle Baseless Info: Implied but unstated claims
- Subtle Conflict: Nuanced contradictions
"""

import pandas as pd
import random
import re
from typing import Optional, List, Literal
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from tqdm import tqdm


# RAGTruth-style hallucination injection prompts
HALLUCINATION_PROMPTS = {
    "Evident Conflict": """Modify the answer to contain a DIRECT CONTRADICTION of a fact in the reference.
Mark the contradicting part with <hal>...</hal> tags.

Reference excerpt:
{reference}

Original answer:
{answer}

Return ONLY the modified answer with <hal> tags around the contradicting part. No explanation.""",

    "Evident Baseless Info": """Add a FABRICATED fact that is NOT in the reference but sounds plausible.
Mark the fabricated part with <hal>...</hal> tags.

Reference excerpt:
{reference}

Original answer:
{answer}

Return ONLY the modified answer with <hal> tags around the fabricated part. No explanation.""",

    "Subtle Baseless Info": """Add an IMPLIED claim that goes beyond what the reference states.
Mark the implied claim with <hal>...</hal> tags.

Reference excerpt:
{reference}

Original answer:
{answer}

Return ONLY the modified answer with <hal> tags around the implied claim. No explanation.""",

    "Subtle Conflict": """Modify the answer to contain a NUANCED CONTRADICTION - something that seems consistent
but actually conflicts with the reference upon careful reading.
Mark the conflicting part with <hal>...</hal> tags.

Reference excerpt:
{reference}

Original answer:
{answer}

Return ONLY the modified answer with <hal> tags around the conflicting part. No explanation.""",
}


@OPERATOR_REGISTRY.register()
class HallucinationInjectionOperator(OperatorABC):
    """Inject RAGTruth-style hallucinations into answers.
    
    This operator takes QA pairs with reference context and injects
    controlled hallucinations for creating evaluation datasets.
    
    Example:
        >>> from dataflow.operators.hallucination_detection import HallucinationInjectionOperator
        >>> from dataflow.serving import LocalHostLLMAPIServing_vllm
        >>> 
        >>> llm = LocalHostLLMAPIServing_vllm(
        ...     hf_model_name_or_path="Qwen/Qwen2.5-72B-Instruct",
        ...     vllm_server_port=8000,
        ... )
        >>> injector = HallucinationInjectionOperator(
        ...     llm_serving=llm,
        ...     hallucination_ratio=0.5,
        ...     hallucination_types=["Evident Conflict", "Evident Baseless Info"],
        ... )
    """
    
    def __init__(
        self,
        llm_serving: LLMServingABC,
        hallucination_ratio: float = 0.5,
        hallucination_types: Optional[List[str]] = None,
        seed: int = 42,
        max_reference_chars: int = 4000,
    ):
        """Initialize the HallucinationInjectionOperator.
        
        Args:
            llm_serving: LLM serving backend for generating hallucinations.
            hallucination_ratio: Fraction of samples to inject hallucinations (0-1).
            hallucination_types: List of hallucination types to use.
                Options: "Evident Conflict", "Evident Baseless Info",
                         "Subtle Baseless Info", "Subtle Conflict"
            seed: Random seed for reproducibility.
            max_reference_chars: Maximum characters from reference to include in prompt.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.hallucination_ratio = hallucination_ratio
        self.hallucination_types = hallucination_types or [
            "Evident Conflict",
            "Evident Baseless Info",
        ]
        self.seed = seed
        self.max_reference_chars = max_reference_chars
        self.rng = random.Random(seed)
        
        # Validate hallucination types
        for hal_type in self.hallucination_types:
            if hal_type not in HALLUCINATION_PROMPTS:
                raise ValueError(
                    f"Unknown hallucination type: {hal_type}. "
                    f"Options: {list(HALLUCINATION_PROMPTS.keys())}"
                )
    
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Returns a description of the operator's functionality."""
        if lang == "zh":
            return (
                "HallucinationInjectionOperator 向LLM生成的答案中注入RAGTruth风格的幻觉。",
                "支持的幻觉类型：明显冲突、明显无依据、微妙无依据、微妙冲突。",
                "输出：带有<hal>标记的修改后答案，用于训练幻觉检测模型。",
            )
        else:
            return (
                "HallucinationInjectionOperator injects RAGTruth-style hallucinations into answers.",
                "Supported types: Evident Conflict, Evident Baseless Info, Subtle Baseless Info, Subtle Conflict.",
                "Output: Modified answers with <hal> tags for hallucination detection training.",
            )
    
    def _get_reference_excerpt(self, context: str) -> str:
        """Get a truncated excerpt from the context for the prompt."""
        if len(context) <= self.max_reference_chars:
            return context
        
        # Take beginning and end
        half = self.max_reference_chars // 2
        return context[:half] + "\n...\n" + context[-half:]
    
    def _inject_hallucination(
        self,
        answer: str,
        context: str,
        hal_type: str,
    ) -> Optional[str]:
        """Inject a hallucination into an answer using the LLM."""
        reference = self._get_reference_excerpt(context)
        prompt = HALLUCINATION_PROMPTS[hal_type].format(
            reference=reference,
            answer=answer,
        )
        
        try:
            response = self.llm_serving.generate(prompt)
            if isinstance(response, list):
                response = response[0]
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Hallucination injection failed: {e}")
            return None
    
    def _parse_hal_tags(self, text: str) -> List[dict]:
        """Parse <hal>...</hal> tags to extract span positions."""
        labels = []
        # Remove tags and track positions
        clean_text = text
        for match in re.finditer(r"<hal>(.*?)</hal>", text, re.DOTALL):
            hal_text = match.group(1)
            labels.append({
                "text": hal_text,
                "label": "hallucinated",
            })
        
        # Clean the text
        clean_text = re.sub(r"<hal>(.*?)</hal>", r"\1", text, flags=re.DOTALL)
        
        # Find positions in clean text
        for label in labels:
            start = clean_text.find(label["text"])
            if start >= 0:
                label["start"] = start
                label["end"] = start + len(label["text"])
        
        return labels, clean_text
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "dataframe",
        output_key: str = "hallucinated_dataframe",
        context_field: str = "context",
        answer_field: str = "answer",
    ) -> None:
        """Run the hallucination injection operation.
        
        Args:
            storage: DataFlow storage object.
            input_key: Key for the input dataframe.
            output_key: Key for the output dataframe.
            context_field: Column name for the reference context.
            answer_field: Column name for the answer to modify.
        """
        df = storage.get(input_key)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(df)}")
        
        # Validate required columns
        for col in [context_field, answer_field]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        n_samples = len(df)
        n_to_inject = int(n_samples * self.hallucination_ratio)
        inject_indices = set(self.rng.sample(range(n_samples), n_to_inject))
        
        self.logger.info(
            f"Injecting hallucinations into {n_to_inject}/{n_samples} samples "
            f"({self.hallucination_ratio*100:.0f}%)"
        )
        
        results = []
        stats = {"total": 0, "injected": 0, "failed": 0, "by_type": {}}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Injecting hallucinations"):
            result = row.to_dict()
            result["has_hallucination"] = False
            result["hallucination_type"] = None
            result["labels"] = []
            
            if idx in inject_indices:
                # Select hallucination type
                hal_type = self.rng.choice(self.hallucination_types)
                
                # Inject hallucination
                modified = self._inject_hallucination(
                    answer=row[answer_field],
                    context=row[context_field],
                    hal_type=hal_type,
                )
                
                if modified and "<hal>" in modified:
                    labels, clean_answer = self._parse_hal_tags(modified)
                    result[answer_field] = clean_answer
                    result["has_hallucination"] = True
                    result["hallucination_type"] = hal_type
                    result["labels"] = labels
                    stats["injected"] += 1
                    stats["by_type"][hal_type] = stats["by_type"].get(hal_type, 0) + 1
                else:
                    stats["failed"] += 1
            
            stats["total"] += 1
            results.append(result)
        
        output_df = pd.DataFrame(results)
        
        # Log statistics
        self.logger.info(f"Injection complete: {stats}")
        self.logger.info(
            f"Success rate: {stats['injected']}/{stats['injected']+stats['failed']} "
            f"({stats['injected']/(stats['injected']+stats['failed']+1e-9)*100:.1f}%)"
        )
        
        storage.set(output_key, output_df)

