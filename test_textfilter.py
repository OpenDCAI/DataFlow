import sys
import logging
import argparse
sys.path.append("..")
sys.path.append(".")
import yaml
from dataflow.utils.utils import merge_yaml, get_processor
# from dataflow.generator.algorithms.CodeFilter import CodeFilter
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="args list")
    parser.add_argument("--yaml_path", type=str, required=True, help="yaml file path")
    args = parser.parse_args()
    
    yaml_path = args.yaml_path
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config = merge_yaml(config)
    algorithm = get_processor("QuratingFilter", config)
    algorithm.run()
    