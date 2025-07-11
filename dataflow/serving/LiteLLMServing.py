import os
import time
import re
from typing import List, Optional, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.core import LLMServingABC
from dataflow.logger import get_logger

try:
    import litellm
except ImportError:
    raise ImportError(
        "litellm is not installed. Please install it with: "
        "pip install open-dataflow[litellm] or pip install litellm"
    )


class LiteLLMServing(LLMServingABC):
    """
    LiteLLM-based serving class that provides unified interface for multiple LLM providers.
    Supports OpenAI, Anthropic, Cohere, Azure, AWS Bedrock, Google and many more providers.
    
    This implementation avoids global state pollution by passing configuration parameters
    directly to each litellm.completion() call, ensuring thread safety and proper isolation
    between different instances. Configuration parameters are immutable after initialization.
    
    Doc: https://docs.litellm.ai/docs/providers
    """
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 key_name_of_api_key: str = "OPENAI_API_KEY",
                 api_base: Optional[str] = None,
                 api_version: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 top_p: float = 1.0,
                 max_workers: int = 10,
                 timeout: int = 60,
                 **kwargs: Any):
        """
        Initialize LiteLLM serving instance.
        
        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-sonnet", "command-r-plus")
            key_name_of_api_key: Environment variable name for API key (default: "OPENAI_API_KEY")
            api_base: Custom API base URL
            api_version: API version for providers that support it
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            max_workers: Number of concurrent workers for batch processing
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to litellm.completion()
            
        Note:
            All configuration parameters are immutable after initialization.
            If you need different settings, create a new instance.
        """
        
        self.model = model
        self.api_base = api_base
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_workers = max_workers
        self.timeout = timeout
        self.kwargs = kwargs
        self.logger = get_logger()
        
        # Get API key from environment variable
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key before using LiteLLMServing."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.key_name_of_api_key = key_name_of_api_key
        
        # Validate model by making a test call
        self._validate_setup()
        
        self.logger.info(f"LiteLLMServing initialized with model: {model}")
    
    def switch_model(self, 
                     model: str,
                     key_name_of_api_key: Optional[str] = None,
                     api_base: Optional[str] = None,
                     api_version: Optional[str] = None,
                     **kwargs: Any):
        """
        Switch to a different model with potentially different API configuration.
        
        Args:
            model: Model name to switch to
            key_name_of_api_key: New environment variable name for API key (optional)
            api_base: New API base URL (optional)
            api_version: New API version (optional)
            **kwargs: Additional parameters for the new model
        """
        # Update model
        self.model = model
        
        # Update API key if new environment variable provided
        if key_name_of_api_key is not None:
            self.api_key = os.environ.get(key_name_of_api_key)
            if self.api_key is None:
                error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key before switching model."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.key_name_of_api_key = key_name_of_api_key
        
        # Update other API configuration if provided
        if api_base is not None:
            self.api_base = api_base
        if api_version is not None:
            self.api_version = api_version
        
        # Update other parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        
        # Validate the new configuration
        self._validate_setup()
        self.logger.success(f"Switched to model: {model}")
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """
        Format LiteLLM response to include reasoning content in a structured format.
        
        This method handles the standardized LiteLLM response format and extracts
        both the main content and any reasoning_content if available.
        
        Args:
            response: The response dictionary from LiteLLM
            
        Returns:
            Formatted string with think/answer tags if reasoning is present,
            otherwise just the content
        """
        try:
            # Extract the main content
            content = response['choices'][0]['message']['content']
            
            # Check if content already has think/answer format
            if re.search(r'<think>.*</think>.*<answer>.*</answer>', content, re.DOTALL):
                return content
            
            # Try to extract reasoning_content from LiteLLM standardized format
            reasoning_content = ""
            try:
                # LiteLLM provides reasoning_content in the message object
                message = response['choices'][0]['message']
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                elif isinstance(message, dict) and 'reasoning_content' in message:
                    reasoning_content = message['reasoning_content']
            except (KeyError, AttributeError):
                pass
            
            # Format the response based on whether reasoning content exists
            if reasoning_content:
                return f"<think>{reasoning_content}</think>\n<answer>{content}</answer>"
            else:
                return content
                
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error formatting response: {e}")
            # Return original response as string if formatting fails
            return str(response)
    
    def _validate_setup(self):
        """Validate the model and API configuration."""
        try:
            # Prepare completion parameters
            completion_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "timeout": self.timeout
            }
            
            # Add optional parameters if provided
            if self.api_key:
                completion_params["api_key"] = self.api_key
            if self.api_base:
                completion_params["api_base"] = self.api_base
            if self.api_version:
                completion_params["api_version"] = self.api_version
                
            # Make a minimal test call to validate setup
            response = litellm.completion(**completion_params)
            self.logger.success("LiteLLM setup validation successful")
        except Exception as e:
            self.logger.error(f"LiteLLM setup validation failed: {e}")
            raise ValueError(f"Failed to validate LiteLLM setup: {e}")
    
    def _generate_single(self, user_input: str, system_prompt: str, retry_times: int = 3) -> str:
        """Generate response for a single input with retry logic.
        
        Args:
            user_input: User input text
            system_prompt: System prompt
            retry_times: Number of retry attempts for transient errors
            
        Returns:
            Generated response string
            
        Raises:
            Exception: If generation fails after all retries
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Prepare completion parameters
        completion_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            **self.kwargs
        }
        
        # Add optional parameters if provided
        if self.api_key:
            completion_params["api_key"] = self.api_key
        if self.api_base:
            completion_params["api_base"] = self.api_base
        if self.api_version:
            completion_params["api_version"] = self.api_version
        
        last_error = None
        for attempt in range(retry_times):
            try:
                response = litellm.completion(**completion_params)
                # Convert response to dict format for format_response
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
                return self.format_response(response_dict)
            except Exception as e:
                last_error = e
                if attempt < retry_times - 1:
                    # Check if error is retryable
                    error_str = str(e).lower()
                    if any(retryable in error_str for retryable in 
                           ["rate limit", "timeout", "connection", "503", "502", "429"]):
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff with max 10s
                        self.logger.warning(f"Retryable error, waiting {wait_time}s: {e}")
                        time.sleep(wait_time)
                        continue
                    
                # Non-retryable error or last attempt
                self.logger.error(f"Error generating response (attempt {attempt + 1}/{retry_times}): {e}")
                break
        
        # Raise the last error instead of returning error string
        raise last_error
    
    def generate_from_input(self, 
                          user_inputs: List[str], 
                          system_prompt: str = "You are a helpful assistant") -> List[str]:
        """
        Generate responses for a list of inputs using concurrent processing.
        
        Args:
            user_inputs: List of user input strings
            system_prompt: System prompt to use for all generations
            
        Returns:
            List of generated responses
        """
        if not user_inputs:
            return []
        
        # Single input case
        if len(user_inputs) == 1:
            try:
                return [self._generate_single(user_inputs[0], system_prompt)]
            except Exception as e:
                # For consistency with batch processing, return error message in list
                error_msg = f"Error: {str(e)}"
                self.logger.error(f"Failed to generate response: {e}")
                return [error_msg]
        
        # Batch processing with threading
        responses = [None] * len(user_inputs)
        
        def generate_with_index(idx: int, user_input: str) -> Tuple[int, str]:
            try:
                response = self._generate_single(user_input, system_prompt)
                return idx, response
            except Exception as e:
                # For batch processing, return error message to maintain list structure
                error_msg = f"Error: {str(e)}"
                self.logger.error(f"Failed to generate response for input {idx}: {e}")
                return idx, error_msg
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(generate_with_index, idx, user_input)
                for idx, user_input in enumerate(user_inputs)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                idx, response = future.result()
                responses[idx] = response
        
        return responses
    
    def generate_embedding_from_input(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Prepare embedding parameters
        embedding_params = {
            "model": self.model,
            "timeout": self.timeout
        }
        
        # Add optional parameters if provided
        if self.api_key:
            embedding_params["api_key"] = self.api_key
        if self.api_base:
            embedding_params["api_base"] = self.api_base
        if self.api_version:
            embedding_params["api_version"] = self.api_version
        
        # Process embeddings with retry logic
        def embed_with_retry(text: str, retry_times: int = 3):
            last_error = None
            for attempt in range(retry_times):
                try:
                    response = litellm.embedding(
                        input=[text],
                        **embedding_params
                    )
                    return response['data'][0]['embedding']
                except Exception as e:
                    last_error = e
                    if attempt < retry_times - 1:
                        error_str = str(e).lower()
                        if any(retryable in error_str for retryable in 
                               ["rate limit", "timeout", "connection", "503", "502", "429"]):
                            wait_time = min(2 ** attempt, 10)
                            self.logger.warning(f"Retryable error in embedding, waiting {wait_time}s: {e}")
                            time.sleep(wait_time)
                            continue
                    self.logger.error(f"Error generating embedding (attempt {attempt + 1}/{retry_times}): {e}")
                    break
            raise last_error
        
        # Process in batches for better performance
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(embed_with_retry, text) for text in texts]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings"):
                try:
                    embedding = future.result()
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"Failed to generate embedding: {e}")
                    # Return empty embedding for failed cases to maintain list structure
                    embeddings.append([])
        
        return embeddings
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for the current provider."""
        try:
            return litellm.model_list
        except Exception as e:
            self.logger.warning(f"Could not retrieve model list: {e}")
            return []
    

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up LiteLLMServing resources")
        # LiteLLM doesn't require explicit cleanup since we don't use global state
        # Instance variables will be garbage collected when the instance is destroyed
        # Clear any references to ensure proper cleanup
        self.api_key = None
        self.kwargs = None


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    
    print("=" * 80)
    print("Testing LiteLLMServing with DeepSeek models")
    print("=" * 80)
    
    # Test 1: Basic single input test with deepseek-chat
    print("\n1. Testing deepseek-chat with single input:")
    print("-" * 40)
    serving = LiteLLMServing(model="deepseek/deepseek-chat", key_name_of_api_key="DEEPSEEK_API_KEY")
    
    test_input = "What is 2 + 2?"
    response = serving.generate_from_input([test_input])
    print(f"Input: {test_input}")
    print(f"Response: {response[0]}")
    print(f"Response type: {type(response[0])}")
    
    # Test 2: Batch input test with deepseek-chat
    print("\n2. Testing deepseek-chat with batch inputs:")
    print("-" * 40)
    batch_inputs = [
        "What is the capital of France?",
        "Explain Python decorators in one sentence.",
        "What is the meaning of life?"
    ]
    responses = serving.generate_from_input(batch_inputs)
    for i, (inp, resp) in enumerate(zip(batch_inputs, responses)):
        print(f"\nBatch {i+1}:")
        print(f"  Input: {inp}")
        print(f"  Response: {resp[:100]}..." if len(resp) > 100 else f"  Response: {resp}")
    
    # Test 3: Custom system prompt
    print("\n3. Testing custom system prompt:")
    print("-" * 40)
    custom_prompt = "You are a helpful math tutor. Answer concisely."
    math_question = "What is the derivative of x^2?"
    response = serving.generate_from_input([math_question], system_prompt=custom_prompt)
    print(f"System prompt: {custom_prompt}")
    print(f"Input: {math_question}")
    print(f"Response: {response[0]}")
    
    # Test 4: Switch to deepseek-r1 model
    print("\n4. Testing model switching to deepseek-r1:")
    print("-" * 40)
    try:
        serving.switch_model("deepseek/deepseek-reasoner", key_name_of_api_key="DEEPSEEK_API_KEY")
        print("Successfully switched to deepseek-r1")
        
        # Test reasoning with deepseek-r1
        reasoning_prompt = "Let's solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?"
        response = serving.generate_from_input([reasoning_prompt])
        print(f"\nInput: {reasoning_prompt}")
        print(f"Response: {response[0]}")
        
        # Check if response contains think/answer tags
        if "<think>" in response[0] and "</think>" in response[0]:
            print("\n✓ Response contains reasoning tags (<think>/<answer>)")
        else:
            print("\n✗ Response does not contain reasoning tags")
            
    except Exception as e:
        print(f"Error switching to deepseek-r1: {e}")
        print("Continuing tests with deepseek-chat...")
    
    # Test 5: Test response format validation
    print("\n5. Testing response format validation:")
    print("-" * 40)
    
    # Test with a complex prompt that might trigger reasoning
    complex_prompt = "Explain why the sky is blue and provide the scientific reasoning behind it."
    response = serving.generate_from_input([complex_prompt])
    print(f"Input: {complex_prompt}")
    print(f"Response length: {len(response[0])} characters")
    print(f"Response preview: {response[0][:150]}...")
    
    # Test 6: Error handling
    print("\n6. Testing error handling:")
    print("-" * 40)
    
    # Test with empty input
    empty_response = serving.generate_from_input([])
    print(f"Empty input response: {empty_response}")
    
    # Test with very long input (might trigger token limit)
    long_input = "Please summarize this: " + "Lorem ipsum " * 500
    response = serving.generate_from_input([long_input[:1000]])  # Truncate for reasonable test
    print(f"Long input test completed: Response length = {len(response[0])} characters")
    
    # Test 7: Model parameters
    print("\n7. Testing different model parameters:")
    print("-" * 40)
    
    # Create a new instance with different parameters
    serving_low_temp = LiteLLMServing(
        model="deepseek/deepseek-chat",
        key_name_of_api_key="DEEPSEEK_API_KEY",
        temperature=0.1,
        max_tokens=50
    )
    
    test_prompt = "Write a haiku about programming"
    response = serving_low_temp.generate_from_input([test_prompt])
    print(f"Low temperature (0.1) response:")
    print(f"Input: {test_prompt}")
    print(f"Response: {response[0]}")
    
    # Cleanup
    print("\n8. Testing cleanup:")
    print("-" * 40)
    serving.cleanup()
    serving_low_temp.cleanup()
    print("✓ Cleanup completed successfully")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    