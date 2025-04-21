from typing import Dict
from argparse import ArgumentError
from jsonargparse import ActionConfigFile, ArgumentParser
from algorithms.QuestionGenerator import QuestionGenerator
from algorithms.QuestionCategoryClassifier import QuestionCategoryClassifier
from algorithms.QuestionDifficultyClassifier import QuestionDifficultyClassifier
import yaml

def init_config(args=None):
    """Initialize new configuration with updated settings."""
    parser = ArgumentParser(default_env=True, default_config_files=None)
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a base config file', required=True)
    parser.add_argument('--input_file', type=str, help='Input file path')
    parser.add_argument('--output_file', type=str, default=None, help='Output file path')
    parser.add_argument('--input_key', type=str, default=None, help='Input prompt key')
    parser.add_argument('--output_key', type=str, help='Output Prompt key')
    parser.add_argument('--algorithm', type=str, default=None, help='Algoritm to run')
    parser.add_argument('--generator_type', type=str, default=None, help='Type of generator, eg. Local model or API')
    parser.add_argument('--configs', type=list, default=None, help='Model configurations')

    try:
        cfg = parser.parse_args(args=args)
        return cfg
    except ArgumentError:
        print('Configuration initialization failed')

import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

def main():
    """
    Main function to initialize configuration and run the pipeline.
    """
    # config = init_config()
    with open("./configs/example.yaml", "r") as f:
        config = yaml.safe_load(f)
    configs = config['configs']
    algorithm = QuestionDifficultyClassifier(configs)
    algorithm.run()

    
if __name__ == '__main__':
    main()