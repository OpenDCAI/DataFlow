import os
import time
import logging
import json
from mimetypes import guess_type
from pathlib import Path
from typing import Any, List, Optional, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.core import LLMServingABC

from tqdm import tqdm
from pydantic import BaseModel

# --- Dependency: Google Vertex AI SDK ---
# Make sure to install the required library:
# pip install "google-cloud-aiplatform>=1.55" pydantic tqdm "pydantic-core<2"
try:
    # NEW: Correct imports for the modern Vertex AI SDK
    import vertexai
    from vertexai.generative_models import (
        GenerativeModel,
        Part,
        Tool,
        FunctionDeclaration,
        GenerationConfig,
        GenerationResponse,
    )
    from google.api_core import exceptions as google_exceptions

except ImportError:
    raise ImportError(
        "Google Cloud AI Platform library not found or is outdated. "
        "Please run 'pip install \"google-cloud-aiplatform>=1.55\" pydantic tqdm'"
    )
# --- Gemini Client Logic (Updated for modern Vertex AI SDK) ---
class GeminiVertexAIClient:
    def __init__(self, project: Optional[str] = None, location: str = 'us-central1'):
        """Initialize Gemini client for Vertex AI."""
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set for Vertex AI authentication.")
        
        vertexai.init(project=project, location=location)
        self.default_model_name = "gemini-2.5-flash"
        # NOTE: We remove the model instance cache because each model instance will now be
        #       tied to a specific system_instruction, which is dynamic.

    def _prepare_content(self, content: Union[str, Path]) -> List[Part]:
        """Prepares content for the Gemini model. Always returns a list of Parts."""
        if isinstance(content, (str, Path)) and os.path.exists(content) and os.path.isfile(content):
            mime_type, _ = guess_type(str(content))
            if not mime_type:
                mime_type = "application/octet-stream"
            # Using from_uri is generally more robust for Vertex AI with local files
            return [Part.from_uri(uri=str(content), mime_type=mime_type)]
        elif isinstance(content, str):
            return [Part.from_text(content)]
        else:
            raise ValueError("Only support text (str) or local file path (str or Path) as input.")

    def generate(
        self,
        system_prompt: str,
        content: Union[str, Path],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_schema: Optional[Union[type[BaseModel], dict]] = None,
    ) -> GenerationResponse:
        """Generate response from a Gemini model on Vertex AI."""
        model_name = model or self.default_model_name
        
        # --- MAJOR FIX HERE ---
        # The system_instruction must be passed during the model's initialization,
        # not to the generate_content() method.
        model_instance = GenerativeModel(
            model_name,
            system_instruction=system_prompt
        )

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        tools = None
        if response_schema is not None:
            if isinstance(response_schema, dict):
                # 已经是 JSON Schema
                schema_dict = response_schema
            else:
                # 是 BaseModel，转换成 JSON Schema
                schema_dict = response_schema.model_json_schema()

            function_declaration = FunctionDeclaration(
                name="extract_data",
                description=f"Extracts structured data according to the provided schema.",
                parameters=schema_dict,
            )
            tools = [Tool(function_declarations=[function_declaration])]
            generation_config.response_mime_type = "application/json"

        contents = self._prepare_content(content)
        
        # --- MAJOR FIX HERE ---
        # Remove the 'system_instruction' argument from this call.
        response = model_instance.generate_content(
            contents=contents,
            generation_config=generation_config,
            tools=tools,
        )
        return response

# --- Main Implementation: GeminiLLMServing ---
class APIGoogleVertexAIServing(LLMServingABC):
    """
    LLM Serving class for Google's Gemini models via Vertex AI API.
    """
    def __init__(self, 
                 model_name: str = "gemini-2.5-flash",
                 project: Optional[str] = None,
                 location: str = 'us-central1',
                 max_workers: int = 10,
                 max_retries: int = 5,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            self.client = GeminiVertexAIClient(project=project, location=location)
            self.logger.info(f"GeminiVertexAIClient initialized successfully for model '{self.model_name}'.")
        except ValueError as e:
            self.logger.error(f"Failed to initialize GeminiVertexAIClient: {e}")
            raise

    def start_serving(self) -> None:
        self.logger.info("GeminiLLMServing: Using Google Cloud API, no local service to start.")

    def cleanup(self) -> None:
        self.logger.info("GeminiLLMServing: No specific cleanup actions needed for API-based client.")

    def load_model(self, model_name_or_path: str, **kwargs: Any):
        self.logger.info(f"Switching model from '{self.model_name}' to '{model_name_or_path}'.")
        self.model_name = model_name_or_path

    def _generate_single_with_retry(self, index: int, user_input: str, system_prompt: str, response_schema: Optional[Union[type[BaseModel], dict]] = None) -> tuple[int, Optional[str]]:
        """Generates a response for a single input with a retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.generate(
                    system_prompt=system_prompt,
                    content=user_input,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_schema=response_schema,
                )
                
                # NEW: Robust response parsing for both text and function calls
                if not response.candidates:
                    finish_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
                    self.logger.warning(
                        f"Request {index} was blocked or produced no candidates. Reason: {finish_reason}. Attempt {attempt + 1}/{self.max_retries}."
                    )
                    if attempt == self.max_retries - 1:
                        return index, f"Error: Content blocked by API. Reason: {finish_reason}"
                    time.sleep(2 ** attempt)
                    continue

                candidate = response.candidates[0]
                
                # Check for safety blocks or other stop reasons
                if candidate.finish_reason.name not in ["STOP", "MAX_TOKENS"]:
                    self.logger.warning(f"Request {index} finished with reason '{candidate.finish_reason.name}'.")

                # Check for function call (structured output)
                if candidate.content.parts and candidate.content.parts[0].function_call:
                    function_call = candidate.content.parts[0].function_call
                    # Convert the structured response to a JSON string
                    result_data = {key: val for key, val in function_call.args.items()}
                    return index, json.dumps(result_data, indent=2)

                # Otherwise, return the plain text response
                return index, response.text

            except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError) as e:
                self.logger.warning(
                    f"API rate limit or server error for request {index} (Attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Request {index} failed after {self.max_retries} retries.")
                    return index, f"Error: API request failed after multiple retries. Details: {e}"
                time.sleep(2 ** attempt)
            
            except Exception as e:
                self.logger.error(f"An unexpected error occurred for request {index} (Attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return index, f"Error: An unexpected error occurred. Details: {e}"
                time.sleep(2 ** attempt)
        
        return index, None

    def generate_from_input(self, user_inputs: List[str], system_prompt: str="", response_schema: Optional[Union[type[BaseModel], dict]] = None) -> List[str]:
        """Generates responses for a list of user inputs in parallel."""
        responses = [None] * len(user_inputs)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._generate_single_with_retry, i, user_input, system_prompt, response_schema): i
                for i, user_input in enumerate(user_inputs)
            }
            
            progress = tqdm(as_completed(future_to_index), total=len(user_inputs), desc="Generating with Gemini on Vertex AI")
            for future in progress:
                index, result = future.result()
                responses[index] = result
        
        return responses

# --- Example Usage (Modified) ---
if __name__ == "__main__":
    # IMPORTANT: Before running, ensure you have authenticated with Google Cloud.
    # This is typically done by running `gcloud auth application-default login` in your terminal.
    
    # --- Step 1: Check for Authentication Credentials ---
    # This is a hard requirement for the API to work.
    # export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/private_project.json"
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("=" * 80)
        print("FATAL: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("This is required for Vertex AI authentication.")
        print("Please authenticate with Google Cloud first by running:")
        print("`gcloud auth application-default login`")
        print("=" * 80)
        exit(1) # Exit if authentication is not configured.

    # --- Step 2: Determine Project ID (Optional) ---
    # The SDK can often auto-discover the project ID if the environment variable is not set.
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    if not gcp_project_id:
        print("=" * 80)
        print("INFO: GCP_PROJECT_ID environment variable is not set.")
        print("The Vertex AI SDK will attempt to automatically discover the project ID from your environment.")
        print("Ensure you have set a default project via `gcloud config set project YOUR_PROJECT_ID`.")
        print("=" * 80)

    # --- Step 3: Run Generation Tests ---
    try:
        # --- Test Case 1: Normal Text Generation ---
        print("\n--- Starting Test 1: Normal Text Generation ---")
        gemini_server_text = APIGoogleVertexAIServing(
            project=gcp_project_id, # Pass the project_id (can be None)
            location='us-central1',
            model_name="gemini-2.5-flash",
            max_workers=5,
            max_retries=3
        )
        system_prompt_text = "You are a helpful assistant that provides concise and accurate answers."
        user_prompts_text = [
            "What is the capital of France?",
            "Write a short poem about the moon.",
            "Explain the concept of photosynthesis in one sentence.",
        ]
        results_text = gemini_server_text.generate_from_input(user_prompts_text, system_prompt_text)
        print("--- Generation Complete ---")
        for i, (prompt, result) in enumerate(zip(user_prompts_text, results_text)):
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Gemini]: {result}")

        # --- Test Case 2: Structured Data Extraction (PyDantic) ---
        print("\n--- Starting Test 2: Structured Data Extraction (JSON Output) ---")
        class UserDetails(BaseModel):
            name: str
            age: int
            city: str

        gemini_server_json =APIGoogleVertexAIServing(
            project=gcp_project_id, # Pass the project_id (can be None)
            location='us-central1',
            model_name="gemini-2.5-flash",
        )
        system_prompt_json = "Extract the user's information from the text and format it as JSON."
        user_prompts_json = [
            "John Doe is 30 years old and lives in New York.",
            "My name is Jane Smith, I am 25, and I reside in London."
        ]
        results_json = gemini_server_json.generate_from_input(user_prompts_json, system_prompt_json, response_schema=UserDetails) # Pass the schema here
        print("--- Generation Complete ---")
        for i, (prompt, result) in enumerate(zip(user_prompts_json, results_json)):
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Gemini JSON]: {result}")
            
        # --- Test Case 3: Structured Data Extraction (Raw JSON Schema) ---
        print("\n--- Starting Test 3: Structured Data Extraction (Raw JSON Schema) ---")
        json_schema = {
            "title": "UserDetails",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }
        gemini_server_json_schema = APIGoogleVertexAIServing(
            project=gcp_project_id, # Pass the project_id (can be None)
            location='us-central1',
            model_name="gemini-2.5-flash",
        )
        system_prompt_json_schema = "Extract the user's information from the text and format it as JSON."
        user_prompts_json_schema = [
            "Alice Johnson is 28 years old and lives in San Francisco.",
            "Bob Brown, aged 35, resides in Toronto."
        ]
        results_json_schema = gemini_server_json_schema.generate_from_input(user_prompts_json_schema, system_prompt_json_schema, response_schema=json_schema)
        print("--- Generation Complete ---")
        for i, (prompt, result) in enumerate(zip(user_prompts_json_schema, results_json_schema)):
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Gemini JSON Schema]: {result}")

    except google_exceptions.PermissionDenied as e:
        print(f"\nERROR: Permission Denied. Details: {e}")
        print("Please ensure your account has the 'Vertex AI User' role on the project.")
        print("Also, verify that the Vertex AI API is enabled for your project.")
    except google_exceptions.NotFound as e:
        print(f"\nERROR: Not Found. Details: {e}")
        print("This might mean the project ID could not be found or the specified model/location is incorrect.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

