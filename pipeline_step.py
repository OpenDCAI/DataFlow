import argparse
from dataflow.utils.utils import pipeline_step, get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, required=True, help="yaml file path")
    parser.add_argument('--step_name', type=str, required=True, help="Processor or generator name")
    parser.add_argument('--step_type', type=str, required=True, help="choose between process and generator")
    args = parser.parse_args()
    logger = get_logger()
    logger.info(f"Start running pipeline step {args.step_name}, using yaml path {args.yaml_path}")
    pipeline_step(**vars(args))