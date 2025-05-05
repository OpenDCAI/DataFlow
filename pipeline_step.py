import argparse
import logging
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(filename)-20s- %(module)-20s- %(funcName)-20s- %(lineno)5d - %(name)-10s | %(levelname)8s | Processno %(process)5d - Threadno %(thread)-15d : %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
    )
from dataflow.utils.utils import pipeline_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, required=True, help="yaml file path")
    parser.add_argument('--step_name', type=str, required=True, help="Processor or generator name")
    parser.add_argument('--step_type', type=str, required=True, help="choose between process and generator")
    args = parser.parse_args()
    logging.info(f"Start running pipeline step {args.step_name}, using yaml path {args.yaml_path}")
    pipeline_step(**vars(args))