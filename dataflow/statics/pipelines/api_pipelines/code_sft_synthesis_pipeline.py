from dataflow.operators.code import (
    InstructionSynthesizer,
    CodeGenerator,
    PairScorer,
    ScoreFilter,
    SandboxValidator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class CodeSFTSynthesis_APIPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/CodePipeline/code_synthesis_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        # Initialize LLM serving for code synthesis
        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )
        
        # Step 1: Code to Instruction synthesizer
        self.instruction_synthesizer_step1 = InstructionSynthesizer(
            llm_serving=self.llm_serving
        )
        
        # Step 2: Instruction to Code generator
        self.code_generator_step2 = CodeGenerator(
            llm_serving=self.llm_serving
        )
        
        # Step 3: Quality scorer for (instruction, code) pairs
        self.pair_scorer_step3 = PairScorer(
            llm_serving=self.llm_serving
        )
        
        # Step 4: Score-based filter
        self.score_filter_step4 = ScoreFilter()
        
        # Step 5: Sandbox validator
        self.sandbox_validator_step5 = SandboxValidator(
            language='python'
        )
    
    def forward(self):
        # Step 1: Generate instructions from raw code
        self.instruction_synthesizer_step1.run(
            storage=self.storage.step(),
            input_code_key="raw_code",
            output_instruction_key="generated_instruction"
        )
        
        # Step 2: Generate code from instructions
        self.code_generator_step2.run(
            storage=self.storage.step(),
            input_instruction_key="generated_instruction",
            output_code_key="generated_code"
        )
        
        # Step 3: Score the generated (instruction, code) pairs
        self.pair_scorer_step3.run(
            storage=self.storage.step(),
            input_instruction_key="generated_instruction",
            input_code_key="generated_code",
            output_score_key="quality_score",
            output_feedback_key="quality_feedback"
        )
        
        # Step 4: Filter out low-quality samples
        self.score_filter_step4.run(
            storage=self.storage.step(),
            input_score_key="quality_score",
            score_threshold=8,
            filter_method="greater_equal"
        )
        
        # Step 5: Validate high-quality code in sandbox
        self.sandbox_validator_step5.run(
            storage=self.storage.step(),
            input_code_key="generated_code",
            output_status_key="sandbox_status",
            output_log_key="sandbox_log"
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = CodeSFTSynthesis_APIPipeline()
    model.forward()
