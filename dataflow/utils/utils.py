import numpy as np
import subprocess
import torch
import json
import yaml


def download_model_from_hf(model_name, model_cache_dir):
    print(f"Downloading {model_name} to {model_cache_dir}.")
    command = ['huggingface-cli', 'download', '--resume-download', model_name, '--local-dir', model_cache_dir]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to download {model_name}.")
        print(result.stderr)
        return False
    print(f"Successfully downloaded {model_name} to {model_cache_dir}.")
    return True

def round_to_sigfigs(num, sigfigs):
    import math
    if isinstance(num, np.float32):
        num = float(num)
    if num == 0:
        return 0
    else:
        return round(num, sigfigs - int(math.floor(math.log10(abs(num)))) - 1)


def recursive_insert(ds_scores_dict, scores: dict, idx_list):
    for k, v in scores.items():
        if isinstance(v, dict):
            recursive_insert(ds_scores_dict[k], v, idx_list)
        elif isinstance(v, torch.Tensor):
            ds_scores_dict[k][idx_list] = v.cpu().detach().numpy()
        elif isinstance(v, np.ndarray):
            ds_scores_dict[k][idx_list] = v
        elif isinstance(v, list):
            ds_scores_dict[k][idx_list] = np.array(v)
        elif isinstance(v, float):
            ds_scores_dict[k][idx_list] = np.array(v)
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")

def recursive_func(scores: dict, func, output: dict):
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive_func(scores[k], func, output[k])
        elif isinstance(v, (np.float64, np.float32, np.ndarray)):
            if isinstance(v, np.ndarray) and np.isnan(v).all():
                output[k] = v
            elif isinstance(v, (np.float64, np.float32)) and np.isnan(v):
                output[k] = v
            else:
                output[k] = func(v)
        elif isinstance(v, str):
            output[k] = v  
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")




def recursive_len(scores: dict):
    import numpy as np
    for _, v in scores.items():
        if isinstance(v, dict):
            return recursive_len(v)
        elif isinstance(v, np.ndarray):
            return v.shape[0]
        elif isinstance(v, list):
            return len(v)
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")
        
def recursive_idx(scores: dict, index, output: dict):
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive_idx(scores[k], index, output[k])
        elif isinstance(v, np.ndarray):
            output[k] = v[index]
        elif isinstance(v, list): 
            output[k] = v[index] 
        else:
            raise ValueError(f"Invalid scores type {type(v)} returned")

def recursive(scores: dict, output: dict):
    for k, v in scores.items():
        if isinstance(v, dict):
            if k not in output.keys():
                output[k] = {}
            recursive(scores[k], output[k])
        else:
            output[k] = v

def list_image_eval_metrics():
    from dataflow.config import init_config
    import pyiqa

    cfg = init_config()
    metric_dict = {}
    metric_dict['image']=pyiqa.list_models(metric_mode="NR")

    for k, v in cfg.image.items():
        if v['data_type'] in metric_dict:
            metric_dict[v['data_type']].append(k)
        else:
            metric_dict[v['data_type']] = [k]
    for k, v in metric_dict.items():
        print(f"metric for {k} data:")
        print(v)


def get_scorer(metric_name, device):
    from dataflow.config import init_config
    from dataflow.utils.registry import MODEL_REGISTRY
    import pyiqa

    cfg = init_config()
    if metric_name in cfg.image:
        model_args = cfg.image[metric_name]
        model_args['model_cache_dir'] = cfg.model_cache_path
        model_args['num_workers'] = cfg.num_workers
        scorer = MODEL_REGISTRY.get(model_args['class_name'])(device=device, args_dict=model_args)
    elif metric_name in pyiqa.list_models(metric_mode="NR"):
        # model_args={}
        model_args = cfg.image['pyiqa']
        model_args['model_cache_dir'] = cfg.model_cache_path
        model_args['num_workers'] = cfg.num_workers
        scorer = MODEL_REGISTRY.get(model_args['class_name'])(device=device, metric_name=metric_name, args_dict=model_args)
    elif metric_name in cfg.video:
        model_args = cfg.video[metric_name]
        scorer = MODEL_REGISTRY.get(metric_name)(model_args)
    else:
        raise ValueError(f"Metric {metric_name} is not supported.")
    
    assert scorer is not None, f"Scorer for {metric_name} is not found."
    return scorer

def new_get_scorer(scorer_name, model_args):
    from dataflow.utils.registry import MODEL_REGISTRY
    print(scorer_name, model_args)
    scorer = MODEL_REGISTRY.get(scorer_name)(args_dict=model_args)
    
    assert scorer is not None, f"Scorer for {scorer_name} is not found."
    return scorer


def calculate_score():
    from ..config import new_init_config
    from dataflow.utils.registry import FORMATTER_REGISTRY
    from dataflow.core import ScoreRecord

    cfg = new_init_config()
    
    # for x in cfg['dependencies']:
    #     if x == 'text':
    #         import dataflow.Eval.Text
    #     elif x == 'image':
    #         import dataflow.Eval.image
    #     elif x == 'video':
    #         import dataflow.Eval.video
    #     else:
    #         raise ValueError('Please Choose Dependencies in text, image, video!')
        

    dataset_dict = {}
    score_record = ScoreRecord()
    for scorer_name, model_args in cfg.scorers.items():
        if "num_workers" in cfg:
            model_args["num_workers"] = cfg.num_workers
        if "model_cache_path" in cfg:
            model_args["model_cache_dir"] = cfg.model_cache_path
        scorer = new_get_scorer(scorer_name, model_args)
        if scorer.data_type not in dataset_dict:
            formatter = FORMATTER_REGISTRY.get(cfg['data'][scorer.data_type]['formatter'])(cfg['data'][scorer.data_type])
            datasets = formatter.load_dataset()
            dataset_dict[scorer.data_type] = datasets
            dataset = datasets[0] if type(datasets) == tuple else datasets
            dataset.set_score_record(score_record)
        else:
            datasets = dataset_dict[scorer.data_type]
        _, score = scorer(datasets)
    save_path = cfg['save_path']
    score_record.dump_scores(save_path)

def get_processor(processor_name, args):
    from dataflow.utils.registry import PROCESSOR_REGISTRY
    print(processor_name, args, flush=True)
    processor = PROCESSOR_REGISTRY.get(processor_name)(args_dict=args)
    
    assert processor is not None, f"Processor for {processor} is not found."
    return processor

def process():
    from ..config import api_init_config
    from dataflow.data import DataFlowDSDict
    from dataflow.utils.registry import FORMATTER_REGISTRY
    from dataflow.core import ScoreRecord
    cfg = api_init_config()
    dataset_dict = DataFlowDSDict()

    if isinstance(cfg.yaml, str):
        with open(cfg.yaml, 'r') as f:
            cfg.yaml = yaml.safe_load(f)  # 解析成字典
    
    for scorer_name, args in cfg.yaml.items():
        if "num_workers" in cfg:
            args["num_workers"] = cfg.num_workers
        if "model_cache_path" in cfg:
            args["model_cache_dir"] = cfg.model_cache_path
        processor = get_processor(scorer_name, args)
        if processor.data_type not in dataset_dict.keys():
            formatter = FORMATTER_REGISTRY.get('TextFormatter')(cfg['data'], cfg['key'], cfg['sft_single_round'], cfg['sft_multi_round'], cfg['RLHF'])
            datasets = formatter.load_dataset()
            dataset_dict[processor.data_type] = datasets
            recorder = range(len(datasets))
            result = np.zeros(len(datasets), dtype=bool)
        else:
            datasets = dataset_dict[processor.data_type]
        processed_dataset, recorder = processor(datasets, recorder)
        dataset_dict[processor.data_type] = processed_dataset
    # save_path = cfg['save_path']
    # for dataset in dataset_dict.values():
    #     dataset.dump(save_path)
    result[recorder] = True
    result = result.tolist()
    print(json.dumps({"bool": result}))
