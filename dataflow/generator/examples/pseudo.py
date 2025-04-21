import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml
from algorithms.PseudoAnswerGenerator import PseudoAnswerGenerator
logging.basicConfig(level=logging.INFO)

def main():
    with open("../configs/pseudo.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # algorithm_name = config['algorithm']
    configs = config['configs']
    algorithm = PseudoAnswerGenerator(configs)
    algorithm.run()
    

if __name__ == "__main__":
    main()