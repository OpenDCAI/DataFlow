from dataflow.operators.core_text import PromptedGenerator
from dataflow.serving import LocalModelLLMServing_sglang, LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.wrapper import BatchWrapper

if __name__ == "__main__":
    
    storage = FileStorage(
        # first_entry_file_name="../example_data/GeneralTextPipeline/translation.jsonl",
        first_entry_file_name="./dataflow/example/GeneralTextPipeline/translation.jsonl",
        cache_path="./cache/temp0_2_topp0_9",
        file_name_prefix="translation",
        cache_type="json",
    )
    llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path="/data0/public_models/Qwen2.5-VL-7B-Instruct",
            sgl_dp_size=1,  # data parallel size
            sgl_tp_size=1,  # tensor parallel size
            sgl_mem_fraction_static=0.8,
    )
    # llm_serving = LocalModelLLMServing_vllm(
    #     hf_model_name_or_path="/data0/public_models/Qwen2.5-VL-7B-Instruct"
    # )
    op = PromptedGenerator(
        llm_serving=llm_serving,
        system_prompt="请将后续内容都翻译成中文，不要续写。:\n",
    )

    batched_op = BatchWrapper(op, batch_size=3, batch_cache=True)
    
    batched_op.run(
        storage=storage.step(),
        input_key="raw_content",
    )