import json
import os
import random

from loguru import logger


class DatasetLoader:
    @staticmethod
    def load(path: str, num_samples: int = None, seed: int = None):
        if not os.path.exists(path):
            logger.error(f'Data file not found at {path}')
            return []

        with open(path, 'r') as file:
            data = json.load(file)

            if seed is not None:
                random.seed(seed)

            if num_samples >= 0:
                return random.sample(data, num_samples) if num_samples <= len(data) else data

            return data
        
    @staticmethod
    def fixed_load(path: str, num_samples: int = None):
        if not os.path.exists(path):
            logger.error(f'Data file not found at {path}')
            return []

        with open(path, 'r') as file:
            data = json.load(file)

            return data[:num_samples] if num_samples >= 0 else data


if __name__ == "__main__":
    sample_path = 'sample.json'
    data_samples = DatasetLoader.load(sample_path, 1, seed=22)
    # print(data_samples)
    for sample in data_samples:
        print(sample)