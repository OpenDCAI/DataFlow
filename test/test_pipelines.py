import pytest
from test_reasoning_math import ReasoningPipeline
from test_reasoning_pretrain import ReasoningPipeline_Pretrain
# from test_pt_filter import PTTextPipeline
from test_agentic_rag import AgenticRAGPipeline
from test_text2sql import Text2SQLPipeline
@pytest.fixture(scope="session")
def llm_serving():
    from dataflow.serving import LocalModelLLMServing_vllm
    return LocalModelLLMServing_vllm(
        model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        tensor_parallel_size=4,
        # max_tokens=8192,
        # max_model_len=8192,
        max_tokens=2048,
        max_model_len=2048,
        model_source="local"
    )

@pytest.mark.gpu
def test_reasoning_pipeline(llm_serving): 
    reasoning_pipe = ReasoningPipeline(llm_serving=llm_serving)
    reasoning_pipe.forward()
    
@pytest.mark.gpu
def test_reasoning_pipeline_pretrain(llm_serving):
    reasoning_pipe_pretrain = ReasoningPipeline_Pretrain(llm_serving=llm_serving)
    reasoning_pipe_pretrain.forward()

# @pytest.mark.gpu
# def test_text_pipeline():
#     text_pipe = PTTextPipeline()
#     text_pipe.forward()

@pytest.mark.gpu
def test_agentic_rag_pipeline(llm_serving):
    rag_pipe = AgenticRAGPipeline(llm_serving=llm_serving)
    rag_pipe.forward()

@pytest.mark.gpu
def test_text2sql_pipeline(llm_serving):
    text2sql_pipe = Text2SQLPipeline(llm_serving=llm_serving)
    text2sql_pipe.forward()

