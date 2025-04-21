import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml

from dataflow.generator.algorithms.QuestionDifficultyClassifier import QuestionDifficultyClassifier
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open("/root/workspace/culfjk4p420c73amv510/herunming/DataFlow/dataflow/generator/configs/questiondifficulty.yaml", "r") as f:
        config = yaml.safe_load(f)
    question_difficulty_classifier = QuestionDifficultyClassifier(config)
    question_difficulty_classifier.run()