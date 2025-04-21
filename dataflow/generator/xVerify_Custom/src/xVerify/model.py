import os
import sys
import yaml

from typing import Literal, Tuple

from loguru import logger
from openai import OpenAI
from openai._exceptions import APITimeoutError
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from .prompts import BASE_TEMPLATE


# Define retry strategy parameters
RETRY_TIMES = 30  # Maximum number of retry attempts
WAIT_TIME_UPPER = 30  # Upper bound for exponential backoff
WAIT_TIME_LOWER = 10  # Lower bound for exponential backoff

TIMEOUT = 60  # API request timeout in seconds


class Model:
    """
    A class to interact with a xVerify model, supporting both local and API-based inference.

    Attributes:
        model_name (str): The name of the model.
        model_path_or_url (str): Path or URL to the model.
        inference_mode (Literal["api", "local"]): The mode of inference, either 'api' or 'local'.
        api_key (str, optional): The API key for API requests.
        temperature (float): Sampling temperature for generation (default is 0.1).
        max_tokens (int): Maximum number of tokens to generate (default is 2048).
        top_p (float): Nucleus sampling parameter (default is 0.7).
    """

    def __init__(
        self,
        model_name: str,
        model_path_or_url: str,
        inference_mode: Literal["api", "local"],
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.7
    ):
        """
        Initializes the Model class with the provided parameters.

        Args:
            model_name (str): The name of the xVerify model.
            model_path_or_url (str): Path or URL to the model.
            inference_mode (str): Mode of inference, either 'api' or 'local'.
            api_key (str, optional): The API key for API requests.
            temperature (float, optional): Sampling temperature (default is 0.1).
            max_tokens (int, optional): Maximum number of tokens to generate (default is 2048).
            top_p (float, optional): Nucleus sampling parameter (default is 0.7).

        Raises:
            ValueError: If inference_mode is not 'api' or 'local'.
            ValueError: If temperature is not between 0 and 1.
            ValueError: If max_tokens is less than or equal to 0.
        """

        if inference_mode not in ["api", "local"]:
            raise ValueError("inference_mode must be either 'local' or 'api'")
        
        if not (0 <= temperature <= 1):
            raise ValueError("temperature should be between 0 and 1")

        if max_tokens <= 0:
            raise ValueError("max_tokens should be greater than 0")
        
        self.model_name = model_name
        self.model_path_or_url = model_path_or_url
        self.inference_mode = inference_mode
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def _load_template(self) -> str:
        """
        Loads the base template for the model from the predefined BASE_TEMPLATE.

        Returns:
            str: The template corresponding to the model name.

        Raises:
            KeyError: If the model's template does not exist.
            Exception: For any other unexpected errors.
        """

        try:
            return BASE_TEMPLATE[self.model_name]
            
        except KeyError as e:
            logger.error(f"Base template for model '{self.model_name}' does not exist.")
            raise KeyError(f"Missing template for model '{self.model_name}'")
        except Exception as e:
            logger.exception("Unexpected error while loading the template")
            raise

    def _initialize_local_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Initializes the local model by loading the tokenizer and model.

        Downloads the model if not found locally and initializes it using Huggingface's transformers.

        Returns:
            Tuple[AutoTokenizer, AutoModelForCausalLM]: The tokenizer and model.

        Raises:
            Exception: If there are any issues while loading the model.
        """

        if not os.path.exists(self.model_path_or_url):
            logger.info(
                f"Model not found locally. Downloading model {self.model_name} from Huggingface.")
            os.system(
                f'huggingface-cli download --resume-download IAAR-Shanghai/{self.model_name} --local-dir {self.model_path_or_url}')
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_or_url, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_url,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

        return tokenizer, model
    
    def _request_local(self, prompt: str) -> str:
        """
        Generates a response using the local model by tokenizing the prompt and generating a response.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The generated response from the model.
        """

        base_template = self._load_template()
        formatted_prompt = base_template.format(query=prompt)
        tokenizer, model = self._initialize_local_model()

        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)

        output_ids = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_tokens, 
            temperature=self.temperature
        )

        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()

    @retry(wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER), stop=stop_after_attempt(RETRY_TIMES), reraise=True)
    def _request_api(self, prompt: str) -> str:
        """
        Sends a request to the API to generate a response using the specified model.

        Args:
            prompt (str): The input text to generate a response for.

        Returns:
            str: The response generated by the API.

        Raises:
            APITimeoutError: If the API request times out.
            Exception: For any other exceptions during the request.
        """

        try:
            model = OpenAI(
                base_url=self.model_path_or_url,
                api_key=self.api_key
            )

            response_obj = model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                temperature = self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                timeout=TIMEOUT
            )

            return response_obj.choices[0].message.content
        except APITimeoutError as e:
            logger.warning(f"Request timed out: {repr(e)}")
            return ''
        except Exception as e:
            logger.warning(repr(e))
            raise

    def request(self, prompt: str) -> str:
        """
        Generates a response based on the given prompt using either the local model or an API call.

        Args:
            prompt (str): The input text to be processed by the model.

        Returns:
            str: The generated response.
        """

        if self.inference_mode == 'api':
            response = self._request_api(prompt)
        else:
            response = self._request_local(prompt)
        
        return response
