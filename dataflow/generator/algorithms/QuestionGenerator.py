import random
import os
import pandas as pd
try:
    from utils import LocalModelGenerator, APIGenerator_aisuite, APIGenerator_request
    from utils.Prompts import QuestionSynthesisPrompt
except ImportError:
    from dataflow.generator.utils import LocalModelGenerator, APIGenerator_aisuite, APIGenerator_request
    from dataflow.generator.utils.Prompts import QuestionSynthesisPrompt
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class QuestionGenerator():
    def __init__(self, config):
        """
        Initialize the QuestionGenerator with the provided configuration.
        """
        self.config = config
        self.prompts = QuestionSynthesisPrompt()

        # Ensure the necessary configuration keys are provided
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_key = self.config.get("input_key", "question")  # default key for question input
        self.num_prompts = self.config.get("num_prompts", 1)  # default number of prompts to use for generation
        # check if num_prompts is a valid number
        if self.num_prompts not in [0,1,2,3,4,5]:
            raise ValueError("num_prompts must be 0, 1, 2, 3, 4, or 5")

        # Validate that input_file and output_file are provided
        if not self.input_file or not self.output_file:
            raise ValueError("Both input_file and output_file must be specified in the config.")

        # Initialize the model
        self.model = self.__init_model__()

    def __init_model__(self):
        """
        Initialize the model generator based on the configuration.
        """
        generator_type = self.config.get("generator_type", "local").lower()

        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions based on num_prompts.
        """
        # Check if input_key is in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {key_list}")

        # Predefined transformation options for diversity
        diversity_mode = [
            "1, 2, 3",
            "1, 2, 4",
            "1, 2, 5",
            "1, 4, 5",
            "1, 2, 3, 4, 5"
        ]

        formatted_prompts = []
        for question in dataframe[self.input_key]:
            if self.num_prompts == 0:
                formatted_prompts.append("")  # Skip generating for this question
            else:
                # Randomly choose the required number of transformations from diversity_mode
                selected_items = random.sample(diversity_mode, self.num_prompts)
                for selected_item in selected_items:
                    used_prompt = self.prompts.question_synthesis_prompt(selected_item, question)
                    formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self):
        """
        Run the question generation process.
        """
        try:
            
            # Read the input file (jsonl format only)
            dataframe = pd.read_json(self.input_file, lines=True)
            if self.input_key not in dataframe.columns:
                raise ValueError(f"input_key: {self.input_key} not found in the dataframe. Available keys: {dataframe.columns.tolist()}")
            if "Synth_or_Input" in dataframe.columns:
                raise ValueError(f"Synth_or_Input is a reserved column name to show if the question is generated or not, please rename it")

            if self.num_prompts == 0:
                print(f"num_prompts is 0, skip generation")
                dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
                print(f"Generated questions saved to {self.output_file}")
                return

            # Reformat the prompts for question generation
            formatted_prompts = self._reformat_prompt(dataframe)

            # Generate responses using the model
            responses = self.model.generate_text_from_input(formatted_prompts)

            # 将新生成的问题作为新的行添加到dataframe中，仍然填写到input_key中,这些行的其他列全部为空
            new_rows = pd.DataFrame(columns=dataframe.columns)
            new_rows[self.input_key] = responses
            new_rows["Synth_or_Input"] = "synth"
            dataframe["Synth_or_Input"] = "input"
            dataframe = pd.concat([dataframe, new_rows], ignore_index=True) # ignore_index=True 表示忽略原来的索引
            

            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Save DataFrame to JSON file
            dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

            print(f"Generated questions saved to {self.output_file}")

        except Exception as e:
            print(f"[错误] 处理过程中发生异常: {e}")
