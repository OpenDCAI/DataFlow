import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml
from dataflow.generator.algorithms.LanguageClassifier import LanguageClassifier
logging.basicConfig(level=logging.INFO)

def main():
    with open("/root/workspace/culfjk4p420c73amv510/herunming/DataFlow/dataflow/generator/configs/classifier.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    configs = config['configs']
    algorithm = LanguageClassifier(configs)
    algorithm.run()
    

if __name__ == "__main__":
    main()