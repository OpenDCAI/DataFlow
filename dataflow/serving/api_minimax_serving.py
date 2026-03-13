import json
import warnings
import os
import logging
import time
from ..logger import get_logger
from .api_llm_serving_request import APILLMServing_request


# Available MiniMax chat models
MINIMAX_MODELS = [
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
]


class APIMinimaxServing(APILLMServing_request):
    """MiniMax LLM serving class using the OpenAI-compatible API.

    MiniMax provides OpenAI-compatible chat completion endpoints.
    This class extends APILLMServing_request with MiniMax-specific defaults
    and constraints.

    Key constraints:
        - temperature must be in (0.0, 1.0], zero is not accepted
        - response_format (json_schema) is not supported
        - Base URL (international): https://api.minimax.io/v1
        - Base URL (China mainland): https://api.minimaxi.com/v1

    Supported models:
        - MiniMax-M2.5: Peak performance with 204K context window
        - MiniMax-M2.5-highspeed: Same performance, faster and more agile

    Example::

        from dataflow.serving import APIMinimaxServing

        # export MINIMAX_API_KEY=your-api-key
        serving = APIMinimaxServing(model_name="MiniMax-M2.5")
        responses = serving.generate_from_input(
            user_inputs=["What is 2+2?"],
            system_prompt="You are a helpful assistant."
        )

    For pricing details, see: https://platform.minimax.io/docs
    """

    def __init__(
        self,
        api_url: str = "https://api.minimax.io/v1/chat/completions",
        key_name_of_api_key: str = "MINIMAX_API_KEY",
        model_name: str = "MiniMax-M2.5",
        temperature: float = 1.0,
        max_workers: int = 10,
        max_retries: int = 5,
        connect_timeout: float = 10.0,
        read_timeout: float = 120.0,
        **configs: dict,
    ):
        """Initialize MiniMax serving.

        Args:
            api_url: MiniMax API endpoint. Defaults to the international endpoint.
                Use ``https://api.minimaxi.com/v1/chat/completions`` for
                China mainland.
            key_name_of_api_key: Environment variable name for the API key.
            model_name: Model to use. One of ``MiniMax-M2.5`` or
                ``MiniMax-M2.5-highspeed``.
            temperature: Sampling temperature. Must be in (0.0, 1.0].
                MiniMax does not accept 0.
            max_workers: Number of concurrent request workers.
            max_retries: Maximum number of retries per request.
            connect_timeout: Connection timeout in seconds.
            read_timeout: Read timeout in seconds.
            **configs: Additional parameters forwarded to the API payload.
        """
        # Validate temperature: MiniMax requires (0.0, 1.0]
        if temperature <= 0.0 or temperature > 1.0:
            raise ValueError(
                f"MiniMax requires temperature in (0.0, 1.0], got {temperature}. "
                "Use a small positive value like 0.01 instead of 0."
            )

        super().__init__(
            api_url=api_url,
            key_name_of_api_key=key_name_of_api_key,
            model_name=model_name,
            temperature=temperature,
            max_workers=max_workers,
            max_retries=max_retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            **configs,
        )

    def _api_chat_with_id(
        self,
        id: int,
        payload,
        model: str,
        is_embedding: bool = False,
        json_schema: dict = None,
    ):
        """Override to strip response_format which MiniMax does not support."""
        if json_schema is not None:
            self.logger.warning(
                "MiniMax does not support response_format / json_schema. "
                "Ignoring json_schema parameter. Use prompt engineering to "
                "request structured output instead."
            )
        return super()._api_chat_with_id(
            id=id,
            payload=payload,
            model=model,
            is_embedding=is_embedding,
            json_schema=None,
        )
