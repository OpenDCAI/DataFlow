from dataflow.utils.registry import PROCESSOR_REGISTRY, FORMATTER_REGISTRY
from dataflow.core import ScoreRecord
from dataflow.config import new_init_config

cfg = new_init_config()
dataset_dict = {}
score_record = ScoreRecord()
for processor_name, processor_args in cfg.processors.items():
    if "num_workers" in cfg:
        processor_args["num_workers"] = cfg.num_workers
    if "model_cache_path" in cfg:
        processor_args["model_cache_dir"] = cfg.model_cache_path
    processor = PROCESSOR_REGISTRY.get(processor_name)(args_dict=processor_args)
    if processor.data_type not in dataset_dict:
        formatter = FORMATTER_REGISTRY.get(cfg['data'][processor.data_type]['formatter'])(cfg['data'][processor.data_type])
        datasets = formatter.load_dataset()
        dataset_dict[processor.data_type] = datasets
        dataset = datasets[0] if type(datasets) == tuple else datasets
        dataset.set_score_record(score_record)
    else:
        datasets = dataset_dict[processor.data_type]    
    dataset_dict[processor.data_type] = processor(dataset_dict[processor.data_type])
    print(dataset_dict[processor.data_type].indices)
