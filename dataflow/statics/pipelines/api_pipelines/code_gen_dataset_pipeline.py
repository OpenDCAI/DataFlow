from dataflow.operators.code import (
    CodeInstructionGenerator,
    CodeCodeToInstructionGenerator,
    CodeInstructionToCodeGenerator,
    CodeQualitySampleEvaluator,
    CodeQualityScoreFilter,
    CodeSandboxSampleEvaluator,
    CodeEnhancementInstructionGenerator
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.core import LLMServingABC

class CodeSFTSynthesis_APIPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/BERT_TRAINING_SERVICE/platform/dataset/inclusionAI/Ling-Coder-SFT/main/data/test.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        # use API server as LLM serving
        self.llm_serving = APILLMServing_request(
            api_url="https://aigc.sankuai.com/v1/openai/native/chat/completions",
            model_name="gpt-4o-2024-11-20",
            max_workers=10
        )
        
        # Step 1: Code to Instruction synthesizer
        self.instruction_generater_step1 = CodeEnhancementInstructionGenerator(
            llm_serving=self.llm_serving
        )

        # Step 2: Instruction to Code generator
        self.code_generator_step2 = CodeInstructionToCodeGenerator(
            llm_serving=self.llm_serving
        )

        # Step 3: Quality evaluator for (instruction, code) pairs
        self.pair_evaluator_step3 = CodeQualitySampleEvaluator(
            llm_serving=self.llm_serving
        )
        
        # Step 4: Score-based filter
        
        self.score_filter_step4 = CodeQualityScoreFilter(
            llm_serving=self.llm_serving,
            min_score=7.0,
            max_score=10.0
        )
        
        # Step 4: Sandbox evaluator
        self.sandbox_evaluator_step5 = CodeSandboxSampleEvaluator(
            language='python'
        )
    
    def forward(self):
        # Step 1: Generate instructions from raw code
        self.instruction_generater_step1.run(
            storage=self.storage.step(),
            input_key="messages",
            output_key="generated_instruction"
        )
        
        # Step 2: Generate code from instructions
        self.code_generator_step2.run(
            storage=self.storage.step(),
            input_key="generated_instruction",
            output_key="generated_code"
        )

        # Step 2: Evaluate the generated (instruction, code) pairs
        self.pair_evaluator_step3.run(
            storage=self.storage.step(),
            input_instruction_key="generated_instruction",
            input_code_key="generated_code" 
        )
        
        # Step 4: Filter out low-quality samples
        self.score_filter_step4.run(
            storage=self.storage.step(),
            input_key="messages",
            output_key="quality_score_filter_label"
        )

        # Step 5: Evaluate high-quality code in sandbox
        self.sandbox_evaluator_step5.run(
            storage=self.storage.step(),
            input_key="generated_code"
        )

if __name__ == "__main__":
    model = CodeSFTSynthesis_APIPipeline()
    model.forward()