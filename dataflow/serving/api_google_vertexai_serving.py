import os
from pyexpat import model
import time
import logging
import json
import re
import uuid
import tempfile
from mimetypes import guess_type
from pathlib import Path
from typing import Any, List, Optional, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.core import LLMServingABC
import fsspec

from tqdm import tqdm
from pydantic import BaseModel
import pandas as pd

# --- Dependency: Google Vertex AI SDK ---
# Make sure to install the required library:
# pip install "google-cloud-aiplatform>=1.55" pydantic tqdm "pydantic-core<2" google-cloud-bigquery google-genai
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
    from google.cloud import bigquery
    
    # For batch processing
    from google import genai
    from google.genai.types import CreateBatchJobConfig

except ImportError:
    raise ImportError(
        "Google Cloud AI Platform library not found or is outdated. "
        "Please run 'pip install \"google-cloud-aiplatform>=1.55\" pydantic tqdm google-cloud-bigquery google-genai'"
    )
# --- Gemini Client Logic (Updated for modern Vertex AI SDK) ---
class GeminiVertexAIClient:
    def __init__(self, project: Optional[str] = None, location: str = 'us-central1'):
        """Initialize Gemini client for Vertex AI."""
       # Check required environment variables
        google_app_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        google_cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION")
        google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        google_genai_use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        self.logger = logging.getLogger(__name__)
        # Validate GOOGLE_APPLICATION_CREDENTIALS
        if not google_app_credentials:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
                "Please set it to the path of your service account key file, e.g.: "
                "export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/your/key.json\""
            )
        
        # Check if credentials file exists
        if not os.path.exists(google_app_credentials):
            raise ValueError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {google_app_credentials}. "
                "Please ensure the path is correct."
            )
        
        # Log environment variable status
        if google_cloud_location:
            location = google_cloud_location
            self.logger.info(f"Using GOOGLE_CLOUD_LOCATION from environment: {location}")
        
        if google_cloud_project:
            if project is None:
                project = google_cloud_project
                self.logger.info(f"Using GOOGLE_CLOUD_PROJECT from environment: {project}")
            else:
                self.logger.warning(
                    f"Project parameter '{project}' provided, but GOOGLE_CLOUD_PROJECT is also set. "
                    f"Using parameter value '{project}'."
                )
        
        if google_genai_use_vertexai:
            self.logger.info(f"GOOGLE_GENAI_USE_VERTEXAI is set: {google_genai_use_vertexai}")

        vertexai.init(project=project, location=location)

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
        model_name = model
        
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
                 use_batch: bool = False,
                 batch_wait: bool = True,
                 batch_dataset: str = "dataflow_batch",
                 csv_filename: Optional[str] = None,
                 bq_csv_filename: Optional[str] = None,
                 ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.project = project
        self.location = location
        self.use_batch = use_batch
        self.batch_wait = batch_wait
        self.batch_dataset = batch_dataset
        self.csv_filename = csv_filename
        self.bq_csv_filename = bq_csv_filename
        
        try:
            self.client = GeminiVertexAIClient(project=project, location=location)
            self.logger.info(f"GeminiVertexAIClient initialized successfully for model '{self.model_name}'.")
        except ValueError as e:
            self.logger.error(f"Failed to initialize GeminiVertexAIClient: {e}")
            raise
        
        # Initialize BigQuery client for batch processing
        try:
            self.bq_client = bigquery.Client(project=project)
            self.logger.info("BigQuery client initialized successfully.")
        except Exception as e:
            self.logger.warning(f"BigQuery client initialization failed: {e}. Batch processing features may not be available.")
            self.bq_client = None
        
        # Initialize Google GenAI client for batch processing
        try:
            self.genai_client = genai.Client()
            self.logger.info("Google GenAI client initialized successfully.")
        except Exception as e:
            self.logger.warning(f"Google GenAI client initialization failed: {e}. Batch processing features may not be available.")
            self.genai_client = None

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

    def generate_from_input(
        self, 
        user_inputs: List[str], 
        system_prompt: str = "", 
        response_schema: Optional[Union[type[BaseModel], dict]] = None,
        use_batch: Optional[bool] = None,
        batch_wait: Optional[bool] = None,
        batch_dataset: Optional[str] = None,
        csv_filename: Optional[str] = None,
        bq_csv_filename: Optional[str] = None,
    ) -> Union[List[str], str]:
        """
        Generates responses for a list of user inputs.
        
        Args:
            user_inputs: List of user input strings to process.
            system_prompt: System prompt for the model.
            response_schema: Optional Pydantic BaseModel or dict for structured output.
            use_batch: If True, use batch processing via BigQuery. If False, use parallel real-time generation.
            batch_wait: If True (and use_batch=True), wait for batch job to complete and return results.
                        If False, return the batch job name immediately for later retrieval.
            batch_dataset: BigQuery dataset name for batch processing (default: "dataflow_batch").
            csv_filename: Optional CSV filename for batch processing. If None, defaults to "batch_{timestamp}_{batch_id}.csv".
        
        Returns:
            - If use_batch=False: List of generated responses (same length as user_inputs).
            - If use_batch=True and batch_wait=True: List of generated responses from batch job.
            - If use_batch=True and batch_wait=False: Batch job resource name (str) for later retrieval.
        """
        if use_batch is None:
            use_batch = self.use_batch
        if batch_wait is None:
            batch_wait = self.batch_wait
        if batch_dataset is None:
            batch_dataset = self.batch_dataset
        if csv_filename is None:
            csv_filename = self.csv_filename
        if bq_csv_filename is None:
            bq_csv_filename = self.bq_csv_filename

        if use_batch:
            return self._generate_with_batch(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
                response_schema=response_schema,
                wait_for_completion=batch_wait,
                dataset_name=batch_dataset,
                csv_filename=csv_filename,
                bq_csv_filename=bq_csv_filename
            )
        else:
            return self._generate_with_parallel(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
                response_schema=response_schema
            )
    
    def _generate_with_parallel(
        self,
        user_inputs: List[str],
        system_prompt: str,
        response_schema: Optional[Union[type[BaseModel], dict]]
    ) -> List[str]:
        """Internal method: Generates responses using parallel real-time API calls."""
        responses = [None] * len(user_inputs)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._generate_single_with_retry, i, user_input, system_prompt, response_schema): i
                for i, user_input in enumerate(user_inputs)
            }
            
            progress = tqdm(as_completed(future_to_index), total=len(user_inputs), desc="Generating with Gemini (Real-time)")
            for future in progress:
                index, result = future.result()
                responses[index] = result
        
        return responses
    
    def _generate_with_batch(
        self,
        user_inputs: List[str],
        system_prompt: str,
        response_schema: Optional[Union[type[BaseModel], dict]],
        wait_for_completion: bool,
        dataset_name: str,
        csv_filename: Optional[str] = None,
        bq_csv_filename: Optional[str] = None
    ) -> Union[List[str], str]:
        """Internal method: Generates responses using batch processing via BigQuery."""
        if not self.bq_client:
            raise RuntimeError(
                "BigQuery client is not initialized. Cannot use batch processing. "
                "Please ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly."
            )
        
        # Generate CSV filename if not provided
        timestamp = int(time.time())
        if csv_filename is None:
            if bq_csv_filename is None:
                batch_id = str(uuid.uuid4())[:8]
                bq_csv_filename = f"batch_{timestamp}_{batch_id}.csv"
            else:
                # Even if user provides a bq_csv_filename, must prefix with timestamp
                bq_csv_filename = f"{bq_csv_filename}_{timestamp}"
            temp_csv_path = os.path.join(tempfile.gettempdir(), bq_csv_filename)
        else:
            # Always prefix provided csv_filename with timestamp when copying to temp
            base_csv_name = Path(csv_filename).name
            bq_csv_filename = f"{base_csv_name}_{timestamp}" if bq_csv_filename is None else f"{bq_csv_filename}_{timestamp}"
            temp_csv_path = os.path.join(tempfile.gettempdir(), bq_csv_filename)
        
        try:
            # Step 1: Generate CSV for batch processing
            self.logger.info(f"Batch mode: Generating CSV with {len(user_inputs)} inputs...")
            temp_csv_path = os.path.join(tempfile.gettempdir(), csv_filename)
            self.generate_bq_csv(
                csv_filename=temp_csv_path,
                system_prompt=system_prompt,
                user_prompts=user_inputs,
                response_schema=response_schema,
                max_token=self.max_tokens
            )
            
            # Step 2: Upload to BigQuery
            self.logger.info("Batch mode: Uploading to BigQuery...")
            bq_uri = self.create_bq_table(temp_csv_path, dataset_name=dataset_name)
            
            # Step 3: Submit batch prediction job
            self.logger.info("Batch mode: Submitting batch prediction job...")
            
            batch_job_name = self.run_batch_prediction(bq_uri, model=self.model_name)
            
            # Clean up temporary CSV file
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                self.logger.info(f"Cleaned up temporary CSV: {temp_csv_path}")
            
            # Step 4: Wait for completion if requested
            if wait_for_completion:
                self.logger.info("Batch mode: Waiting for batch job to complete...")
                results = self._wait_and_retrieve_batch_results(batch_job_name, len(user_inputs))
                return results
            else:
                self.logger.info(f"Batch job submitted: {batch_job_name}. Use retrieve_batch_results() to get results later.")
                return batch_job_name
        
        except Exception as e:
            # Clean up temporary CSV file on error
            if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            self.logger.error(f"Batch processing failed: {e}")
            raise
    
    def _wait_and_retrieve_batch_results(self, batch_job_name: str, expected_count: int) -> List[str]:
        """
        Internal method: Waits for a batch job to complete and retrieves results.
        
        Args:
            batch_job_name: The resource name of the batch job.
            expected_count: Expected number of results.
        
        Returns:
            List of generated responses.
        """
        if not self.genai_client:
            raise RuntimeError("Google GenAI client is not initialized. Cannot retrieve batch results.")
        
        try:
            # Get the batch job
            batch_job = self.genai_client.batches.get(name=batch_job_name)
            
            # Wait for completion with progress bar
            self.logger.info("Waiting for batch job to complete (this may take several minutes)...")
            
            # Poll for completion
            max_wait_time = 3600 * 1000  # 1000 hours max
            poll_interval = 30  # Check every 30 seconds
            elapsed_time = 0
            
            with tqdm(total=expected_count, desc="Batch job progress") as pbar:
                while elapsed_time < max_wait_time:
                    batch_job = self.genai_client.batches.get(name=batch_job_name)
                    state = batch_job.state
                    
                    if state == "JOB_STATE_SUCCEEDED":
                        pbar.update(100 - pbar.n)
                        self.logger.info("Batch job completed successfully!")
                        break
                    elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                        raise RuntimeError(f"Batch job failed with state: {state}")
                    
                    # Update progress bar (estimate based on time)
                    progress = min(90, int((elapsed_time / max_wait_time) * 100))
                    pbar.update(progress - pbar.n)
                    
                    time.sleep(poll_interval)
                    elapsed_time += poll_interval
                
                if elapsed_time >= max_wait_time:
                    raise TimeoutError(f"Batch job did not complete within {max_wait_time} seconds")

            output_table = batch_job.dest.bigquery_uri.replace("bq://", "")
            project, dataset, table = output_table.split(".")
            table_id = f"{project}.{dataset}.{table}"
            query = f"SELECT * FROM `{table_id}`"
            df = self.bq_client.query(query).to_dataframe()

            results = self._parse_batch_results(df, expected_count)

            return results
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve batch results: {e}")
            raise
    
    def _parse_batch_results(self, df: pd.DataFrame, expected_count: int) -> List[str]:
        """
        Internal method: Parses batch results from DataFrame.
        
        Args:
            df: DataFrame containing batch results.
            expected_count: Expected number of results.
        
        Returns:
            List of generated responses in original order.
        """
        results = [None] * expected_count
        
        for _, row in df.iterrows():
            try:
                # Get the index from the response
                if 'index' in df.columns:
                    idx = int(row['index'])
                else:
                    # Fallback to row number
                    idx = _
                
                # Extract the response text
                if 'response' in df.columns:
                    response_json = json.loads(row['response'])
                    # Navigate through the response structure
                    if 'candidates' in response_json and len(response_json['candidates']) > 0:
                        candidate = response_json['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if parts and 'text' in parts[0]:
                                results[idx] = parts[0]['text']
                            elif parts and 'functionCall' in parts[0]:
                                # Structured output
                                results[idx] = json.dumps(parts[0]['functionCall']['args'])
                
                # Fallback: if we couldn't parse, store the raw response
                if results[idx] is None and 'response' in df.columns:
                    results[idx] = row['response']
            
            except Exception as e:
                self.logger.warning(f"Failed to parse result at index {idx}: {e}")
                results[idx] = f"Error: Failed to parse result"
        
        return results
    
    def retrieve_batch_results(self, batch_job_name: str, expected_count: int) -> List[str]:
        """
        Retrieves results from a previously submitted batch job.
        
        Args:
            batch_job_name: The resource name of the batch job (returned when use_batch=True and batch_wait=False).
            expected_count: Expected number of results.
        
        Returns:
            List of generated responses.
        
        Example:
            # Submit batch job without waiting
            job_name = serving.generate_from_input(inputs, system_prompt, use_batch=True, batch_wait=False)
            
            # Later, retrieve results
            results = serving.retrieve_batch_results(job_name, len(inputs))
        """
        return self._wait_and_retrieve_batch_results(batch_job_name, expected_count)
    
    # --- Batch Processing Methods ---
    
    def create_bq_dataset(self, dataset_name: str = "polymer") -> None:
        """
        Creates a BigQuery dataset if it does not already exist.
        
        Args:
            dataset_name: The name of the dataset to create. Defaults to "polymer".
        
        Raises:
            RuntimeError: If BigQuery client is not initialized.
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client is not initialized. Cannot create dataset.")
        
        dataset_ref = self.bq_client.dataset(dataset_name)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = self.location
        
        try:
            self.bq_client.create_dataset(dataset)
            self.logger.info(f"Dataset '{dataset_name}' created successfully.")
        except Exception as e:
            if "Already Exists" in str(e):
                self.logger.info(f"Dataset '{dataset_name}' already exists.")
            else:
                self.logger.error(f"Failed to create dataset '{dataset_name}': {e}")
                raise
    
    def generate_bq_csv(
        self,
        csv_filename: str,
        system_prompt: str,
        user_prompts: List[str],
        doi_list: Optional[List[str]] = None,
        response_schema: Optional[Union[type[BaseModel], dict]] = None,
        max_token: int = 500
    ) -> str:
        """
        Generates a CSV file for batch processing with the Gemini API.
        
        Args:
            csv_filename: The name of the output CSV file.
            system_prompt: The system prompt string.
            user_prompts: A list of texts to be processed.
            doi_list: Optional list of DOIs corresponding to user prompts.
            response_schema: An optional Pydantic BaseModel class or dict for the response schema.
            max_token: Maximum number of output tokens. Defaults to 500.
        
        Returns:
            The name of the generated CSV file.
        """
        df = pd.DataFrame({"user_prompt": user_prompts})
        
        def create_batch_request_json(row) -> str:
            request_parts = [
                {"text": row["user_prompt"]},
            ]
            
            generation_config = {
                "temperature": self.temperature,
                "maxOutputTokens": max_token,
                "stopSequences": ["\n\n\n\n"],
                "responseLogprobs": True,
                "logprobs": 10,
            }
            
            if response_schema:
                generation_config["responseMimeType"] = "application/json"
                # Handle both BaseModel and dict schemas
                if isinstance(response_schema, dict):
                    generation_config["responseSchema"] = response_schema
                else:
                    generation_config["responseSchema"] = response_schema.model_json_schema()
            else:
                generation_config["responseMimeType"] = "text/plain"
            
            return json.dumps(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": request_parts,
                        },
                    ],
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "generationConfig": generation_config,
                }
            )
        
        df["request"] = df.apply(create_batch_request_json, axis=1)
        if doi_list:
            df["doi"] = doi_list
            df.to_csv(csv_filename, index_label="index", columns=["doi", "request"])
        else:
            df.to_csv(csv_filename, index_label="index", columns=["request"])
        
        self.logger.info(f"Generated batch CSV file: {csv_filename}")
        return csv_filename
    
    def create_bq_table(self, csv_path: str, dataset_name: str = "polymer") -> str:
        """
        Creates a BigQuery table from a CSV file.
        
        Args:
            csv_path: The path to the CSV file.
            dataset_name: The name of the dataset to create the table in. Defaults to "polymer".
        
        Returns:
            The BigQuery URI of the created table (e.g., "bq://project.dataset.table").
        
        Raises:
            RuntimeError: If BigQuery client is not initialized.
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client is not initialized. Cannot create table.")
        
        self.create_bq_dataset(dataset_name)
        
        table_name = Path(csv_path).stem
        table_id = f"{dataset_name}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            autodetect=True,
            skip_leading_rows=1,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        
        with open(csv_path, "rb") as source_file:
            job = self.bq_client.load_table_from_file(source_file, table_id, job_config=job_config)
        
        job.result()
        
        self.logger.info(f"Loaded {job.output_rows} rows into {table_id}")
        return f"bq://{self.bq_client.project}.{dataset_name}.{table_name}"
    
    def run_batch_prediction(
        self,
        input_bq_uri: str,
        model: str = None,
        output_file_path: str = ""
    ) -> str:
        """
        Runs a batch prediction job using the Gemini API.
        
        Args:
            input_bq_uri: The BigQuery URI of the input table (e.g., "bq://project.dataset.table")
                          or a path to a CSV file. If a CSV path is provided, a BigQuery table
                          will be created from it.
            model: The ID of the model to use for prediction. Defaults to the instance's model_name.
            output_file_path: Optional. The desired path for the output file. If empty, defaults to
                              the input path with "_result" appended to the table name.
        
        Returns:
            The name/resource path of the created batch prediction job.
        
        Raises:
            RuntimeError: If BigQuery client or GenAI client is not initialized.
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client is not initialized. Cannot run batch prediction.")
        
        if not self.genai_client:
            raise RuntimeError("Google GenAI client is not initialized. Cannot run batch prediction.")
        
        # Check if input_bq_uri is a CSV file path
        if input_bq_uri.endswith(".csv") and Path(input_bq_uri).is_file():
            self.logger.info(f"CSV file detected: {input_bq_uri}. Creating BigQuery table...")
            input_bq_uri = self.create_bq_table(input_bq_uri)
            self.logger.info(f"BigQuery table created from CSV: {input_bq_uri}")
        
        # Parse input_bq_uri to get project, dataset, and table name
        match = re.match(r"bq://([^.]+)\.([^.]+)\.(.+)", input_bq_uri)
        if not match:
            raise ValueError(
                f"Invalid input_bq_uri format: {input_bq_uri}. "
                "Must be a BigQuery URI (bq://project.dataset.table) or a valid CSV file path."
            )
        
        project_id = match.group(1)
        dataset_name = match.group(2)
        input_table_name = match.group(3)
        
        # Construct the output BigQuery URI
        if output_file_path:
            output_table_name = Path(output_file_path).stem
        else:
            output_table_name = f"{input_table_name}_result"
        
        output_uri = f"bq://{project_id}.{dataset_name}.{output_table_name}"
        
        # Use the model name from instance or provided parameter
        model_name = model or self.model_name
        
        try:
            # Create batch prediction job using Google GenAI API
            batch_job = self.genai_client.batches.create(
                model=model_name,
                src=input_bq_uri,
                config=CreateBatchJobConfig(dest=output_uri)
            )
            
            self.logger.info(f"Batch job '{batch_job.name}' created. State: {batch_job.state}")
            return batch_job.name
        
        except Exception as e:
            self.logger.error(f"Failed to create batch prediction job: {e}")
            raise
    
    def download_bq_table(self, table_id: str, output_file_path: str = "") -> str:
        """
        Downloads data from a BigQuery table to a CSV file.
        
        Args:
            table_id: The full BigQuery table ID (e.g., "project.dataset.table").
            output_file_path: Optional. The path to save the downloaded CSV file.
                              If empty, defaults to the table name with a .csv extension.
        
        Returns:
            The path to the downloaded CSV file.
        
        Raises:
            RuntimeError: If BigQuery client is not initialized.
        """
        if not self.bq_client:
            raise RuntimeError("BigQuery client is not initialized. Cannot download table.")
        
        # Parse table_id to get project, dataset, and table name
        parts = table_id.split('.')
        if len(parts) != 3:
            raise ValueError(
                f"Invalid table_id format: {table_id}. Expected 'project.dataset.table'."
            )
        
        table_name = parts[2]
        
        if not output_file_path:
            output_file_path = f"{table_name}.csv"
        
        try:
            table = self.bq_client.get_table(table_id)
            rows = self.bq_client.list_rows(table)
            df = rows.to_dataframe()
            df.to_csv(output_file_path, index=False)
            
            self.logger.info(f"Downloaded {rows.total_rows} rows from {table_id} to {output_file_path}")
            return output_file_path
        
        except Exception as e:
            self.logger.error(f"Error downloading BigQuery table {table_id}: {e}")
            raise

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
        
        # --- Test Case 4: Batch Processing with BigQuery (Submit without waiting) ---
        print("\n--- Starting Test 4: Batch Processing (Submit without waiting) ---")
        from typing import Literal
        
        class Capital(BaseModel):
            capital: Literal["Paris", "Beijing", "London", "Tokyo"]
        
        gemini_server_batch = APIGoogleVertexAIServing(
            project=os.getenv("GCP_PROJECT_ID"),
            location='us-central1',
            model_name="gemini-2.5-flash",
        )
        
        system_prompt_batch = "You are a helpful assistant that answers geography questions."
        user_prompts_batch = [
            "What is the capital of France?",
            "What is the capital of China?",
            "What is the capital of the United Kingdom?",
            "What is the capital of Japan?"
        ]
        
        try:
            # Submit batch job without waiting (batch_wait=False)
            batch_job_name = gemini_server_batch.generate_from_input(
                user_inputs=user_prompts_batch,
                system_prompt=system_prompt_batch,
                response_schema=Capital,
                use_batch=True,
                batch_wait=False  # Don't wait for completion
            )
            print(f"Batch job submitted: {batch_job_name}")
            print("Note: Use retrieve_batch_results(batch_job_name, len(inputs)) to get results later.")
            
        except Exception as e:
            print(f"Batch processing test (no wait) failed: {e}")
        
        # --- Test Case 5: Batch Processing with BigQuery (Wait for completion) ---
        print("\n--- Starting Test 5: Batch Processing (Wait for completion) ---")
        print("WARNING: This test will wait for batch job completion, which may take several minutes.")
        print("Skipping this test in automated runs. Set ENABLE_BATCH_WAIT_TEST=1 to enable.")
        
        try:
            # Submit batch job and wait for completion (batch_wait=True, default)
            results_batch = gemini_server_batch.generate_from_input(
                user_inputs=user_prompts_batch,
                system_prompt=system_prompt_batch,
                response_schema=Capital,
                use_batch=True,
                batch_wait=True  # Wait for completion
            )
            print("--- Batch Generation Complete ---")
            for i, (prompt, result) in enumerate(zip(user_prompts_batch, results_batch)):
                print(f"\n[Prompt {i+1}]: {prompt}")
                print(f"[Gemini Batch]: {result}")
        
        except Exception as e:
            print(f"Batch processing test (with wait) failed: {e}")

    except google_exceptions.PermissionDenied as e:
        print(f"\nERROR: Permission Denied. Details: {e}")
        print("Please ensure your account has the 'Vertex AI User' role on the project.")
        print("Also, verify that the Vertex AI API is enabled for your project.")
    except google_exceptions.NotFound as e:
        print(f"\nERROR: Not Found. Details: {e}")
        print("This might mean the project ID could not be found or the specified model/location is incorrect.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

