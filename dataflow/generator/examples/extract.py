import sys
sys.path.append("..")
sys.path.append(".")
import yaml
from algorithms.AnswerExtraction_qwenmatheval import AnswerExtraction_qwenmatheval

if __name__ == "__main__":
    with open("../configs/extract.yaml", "r") as f:
        config = yaml.safe_load(f)
    AnswerExtraction_qwenmatheval(config).run()