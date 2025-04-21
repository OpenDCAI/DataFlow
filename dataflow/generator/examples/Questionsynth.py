import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml

from algorithms.QuestionGenerator import QuestionGenerator
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open("configs/Questionsynth.yaml", "r") as f:
        config = yaml.safe_load(f)
    question_generator = QuestionGenerator(config)
    question_generator.run()
