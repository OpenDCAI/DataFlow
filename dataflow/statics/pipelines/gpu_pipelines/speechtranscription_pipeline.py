from dataflow.operators.core_speech import Speech2TextGenerator
from dataflow.serving import LocalModelLALMServing_vllm
from dataflow.utils.storage import FileStorage

class SpeechTranscription_GPUPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/SpeechTranscription/pipeline_speechtranscription.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving = LocalModelLALMServing_vllm(
            hf_model_name_or_path='Qwen/Qwen2-Audio-7B-Instruct',
            vllm_tensor_parallel_size=4,
            vllm_max_tokens=8192,
        )
        self.speech_transcriptor = Speech2TextGenerator(
            llm_serving = self.llm_serving,
            system_prompt="你是一个专业的翻译员，你需要将语音转录为文本。"
        )
    
    def forward(self):
        self.speech_transcriptor.run(
            storage=self.storage.step(),
            input_key="raw_content"
        )

if __name__ == "__main__":
    pipeline = SpeechTranscription_GPUPipeline()
    pipeline.forward()