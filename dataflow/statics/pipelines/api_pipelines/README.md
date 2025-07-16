# Dataflow pipelines with API keys

- Note that you have to export your api key to your environment variables before running the code:
```shell
export DF_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- Then you can modify the settings of the files in each pipeline files. Especially the parameters for `llm_serving` and `storage` 
```python
# global storage class
self.storage = FileStorage(
            first_entry_file_name="../example/ReasoningPipeline/pipeline_math_short.json",  # path to the first entry file
            cache_path="./cache_local", # path to store middle results
            file_name_prefix="dataflow_cache_step", # prefix of the cache file name
            cache_type="jsonl", # type of the cache file
        )

# use API server as LLM serving
llm_serving = APILLMServing_request(
        api_url="https://api.openai.com/v1/chat/completions", # url of the API server
        model_name="gpt-4o",
        max_workers=100
)
```

- Then you can run the code:
```shell
python reasoning_pipeline.py
```
