import logging
import aisuite as ai
import pandas as pd

class APIGenerator_aisuite:
    def __init__(self, config: dict):
        configs = config.configs[0]  # Assuming config.configs is a list of configurations

        # Extract the configurations from the provided dictionary
        self.model_id = configs.get("model_id", 'openai:gpt-4o')
        self.temperature = configs.get("temperature", 0.75)
        self.top_p = configs.get("top_p", 1)
        self.max_tokens = configs.get("max_tokens", 20)
        self.n = configs.get("n", 1)
        self.stream = configs.get("stream", False)
        self.stop = configs.get("stop", None)
        self.presence_penalty = configs.get("presence_penalty", 0)
        self.frequency_penalty = configs.get("frequency_penalty", 0)
        self.logprobs = configs.get("logprobs", None)
        self.prompt = configs.get("prompt", "You are a helpful assistant")
        
        # Input and output file paths and keys
        self.input_file = config.get("input_file", None)
        self.output_file = config.get("output_file", None)
        self.input_prompt_key = config.get("input_key", "prompt")
        self.output_text_key = config.get("output_key", "response")

        logging.info(f"API Generator will generate text using {self.model_id}")

    def generate_text(self):
        client = ai.Client()
        models = self.model_id.split(',')
        outputs = []

        # Read input file (accept jsonl file only)
        dataframe = pd.read_json(self.input_file, lines=True)
        
        # Check if input_prompt_key is in the dataframe
        if self.input_prompt_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_prompt_key: {self.input_prompt_key} not found in the dataframe, please check the input_prompt_key: {key_list}")
        
        # Check if output_text_key is in the dataframe
        if self.output_text_key in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_text_key} in the dataframe, which leads to overwriting the existing column, please check the output_text_key: {key_list}")
        
        # Generate text
        user_prompts = dataframe[self.input_prompt_key].tolist()
        for user_prompt in user_prompts:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=self.n,
                stream=self.stream,
                stop=self.stop,
                logprobs=self.logprobs,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
            content = response.choices[0].message.content
            outputs.append(content)
        return outputs

    def generate_and_save(self):
        outputs = self.generate_text()
        dataframe = pd.read_json(self.input_file, lines=True)
        dataframe[self.output_text_key] = outputs
        dataframe.to_json(self.output_file, orient='records', lines=True, force_ascii=False)

    def generate_text_from_input(self, questions: list[str]) -> list[str]:
        client = ai.Client()
        outputs = []
        for question in questions:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": question}
            ]
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=self.n,
                stream=self.stream,
                stop=self.stop,
                logprobs=self.logprobs,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
            )
            content = response.choices[0].message.content
            outputs.append(content)
        return outputs
