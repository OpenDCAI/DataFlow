"""
Example Pipeline for Hallucination Detection Dataset Generation.

This pipeline demonstrates how to use the hallucination detection operators
to create a long-context evaluation dataset.

Usage:
    python example_pipeline.py --api-url http://localhost:8000/v1
"""

import argparse
import pandas as pd
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.serving import LocalHostLLMAPIServing_vllm
from dataflow.operators.hallucination_detection import (
    LongContextFilterOperator,
    HallucinationInjectionOperator,
)
from dataflow.operators.core_text.generate import Text2QAGenerator


class HallucinationDetectionPipeline(PipelineABC):
    """Pipeline for generating hallucination detection evaluation datasets.
    
    Steps:
    1. Filter samples by token length (8K-24K tokens)
    2. Generate QA pairs from long documents
    3. Inject hallucinations into a subset
    4. Output annotated dataset
    """
    
    def __init__(
        self,
        llm_serving,
        min_tokens: int = 8000,
        max_tokens: int = 24000,
        hallucination_ratio: float = 0.5,
    ):
        super().__init__()
        
        # Initialize operators
        self.filter_op = LongContextFilterOperator(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        
        self.inject_op = HallucinationInjectionOperator(
            llm_serving=llm_serving,
            hallucination_ratio=hallucination_ratio,
            hallucination_types=["Evident Conflict", "Evident Baseless Info", "Subtle Baseless Info"],
        )
    
    def forward(self):
        """Define the pipeline flow."""
        # Step 1: Filter by token length
        self.filter_op.run(
            storage=self.storage,
            input_key="raw_data",
            output_key="long_context_data",
        )
        
        # Step 2: Inject hallucinations
        self.inject_op.run(
            storage=self.storage,
            input_key="long_context_data",
            output_key="hallucinated_data",
            context_field="document",
            answer_field="answer",
        )


def main():
    parser = argparse.ArgumentParser(description="Generate hallucination detection dataset")
    parser.add_argument("--api-url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct", help="Model name")
    parser.add_argument("--input-file", required=True, help="Input JSONL file with documents")
    parser.add_argument("--output-file", required=True, help="Output JSONL file")
    parser.add_argument("--min-tokens", type=int, default=8000, help="Minimum tokens")
    parser.add_argument("--max-tokens", type=int, default=24000, help="Maximum tokens")
    parser.add_argument("--hal-ratio", type=float, default=0.5, help="Hallucination ratio")
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input_file}")
    df = pd.read_json(args.input_file, lines=True)
    
    # Initialize LLM serving
    from dataflow.serving.api_llm_serving_request import APILLMServingRequest
    llm = APILLMServingRequest(
        api_url=args.api_url,
        model_name=args.model,
    )
    
    # Create pipeline
    pipeline = HallucinationDetectionPipeline(
        llm_serving=llm,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        hallucination_ratio=args.hal_ratio,
    )
    
    # Create storage and run
    storage = DataFlowStorage()
    storage.set("raw_data", df)
    pipeline.storage = storage
    
    # Compile and run
    pipeline.compile()
    pipeline.forward()
    
    # Get output
    output_df = storage.get("hallucinated_data")
    print(f"Generated {len(output_df)} samples")
    
    # Save
    output_df.to_json(args.output_file, orient="records", lines=True)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()

