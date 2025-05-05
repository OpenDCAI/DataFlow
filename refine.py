import sys
import logging
sys.path.append("..")
sys.path.append(".")
import yaml
from dataflow.generator.algorithms.CodeRefiner import CodeRefiner
logging.basicConfig(level=logging.INFO)

def main():
    with open("/root/workspace/culfjk4p420c73amv510/herunming/DataFlow/dataflow/generator/configs/refine.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # algorithm_name = config['algorithm']
    configs = config['configs']
    algorithm = CodeRefiner(configs)
    algorithm.run()
    

if __name__ == "__main__":
    main()