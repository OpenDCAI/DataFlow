from .api_llm_serving_request import APILLMServing_request


class AtlasCloudLLMServing(APILLMServing_request):
    """Atlas Cloud OpenAI-compatible LLM serving preset.

    Uses ``ATLASCLOUD_API_KEY`` by default and points to the Atlas Cloud
    chat-completions endpoint while keeping the same request/response behavior
    as ``APILLMServing_request``.
    """

    def __init__(
        self,
        api_url: str = "https://api.atlascloud.ai/v1/chat/completions",
        key_name_of_api_key: str = "ATLASCLOUD_API_KEY",
        model_name: str = "qwen/qwen3.5-flash",
        **configs,
    ):
        super().__init__(
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            **configs,
        )
