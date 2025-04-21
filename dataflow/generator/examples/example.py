import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml
from dataflow.generator.algorithms.AnswerGenerator_reasoning import AnswerGenerater_reasoning
logging.basicConfig(level=logging.INFO)

def main():
    with open("configs/reasoning.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    algorithm_name = config['algorithm']
    configs = config['configs']
    algorithm = AnswerGenerater_reasoning(configs)
    algorithm.run()
    

if __name__ == "__main__":
    main()