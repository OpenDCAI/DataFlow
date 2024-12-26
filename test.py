import random
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY, FORMATTER_REGISTRY
from torch.utils.data import DataLoader
from dataflow.core import ScoreRecord
from dataflow.config import new_init_config

cfg = new_init_config()
dataset_dict = {}
score_record = ScoreRecord()
for filter_name, filter_args in cfg.filters.items():
    if "num_workers" in cfg:
        filter_args["num_workers"] = cfg.num_workers
    if "model_cache_path" in cfg:
        filter_args["model_cache_dir"] = cfg.model_cache_path
    filter = PROCESSOR_REGISTRY.get(filter_name)(filter_args)
    if filter.data_type not in dataset_dict:
        formatter = FORMATTER_REGISTRY.get(cfg['data'][filter.data_type]['formatter'])(cfg['data'][filter.data_type])
        datasets = formatter.load_dataset()
        dataset_dict[filter.data_type] = datasets
        dataset = datasets[0] if type(datasets) == tuple else datasets
        dataset.set_score_record(score_record)
    else:
        datasets = dataset_dict[filter.data_type]    
    dataset_dict[filter.data_type] = filter(dataset_dict[filter.data_type])
    print(dataset_dict[filter.data_type].indices)
    # print(score)
    # if isinstance(dataset, tuple):
    #     print(dataset[0].scores_list)
    #     print(dataset[0].meta_score)
    # else:
    #     # print(calculate_statistics(dataset.scores_list, scorer.score_name))
    #     print(dataset.scores_list)
    #     if len(dataset.meta_score.items()) > 0:
    #         print(dataset.meta_score)
# with open('sta.json', 'w') as f:
#     json.dump(score_record.calculate_statistics('DeitaComplexityScore'), f, indent=4)
# for _, dataset in dataset_dict.items():
#     dataset.dump_scores(save_path)


# cfg = init_config()
# formatter = FORMATTER_REGISTRY.get(cfg['data']['video-caption']['formatter'])(cfg['data']['video-caption'])
# dataset = formatter.load_dataset()
# scorer = MODEL_REGISTRY.get('EMScorer')(cfg['video']['EMScorer'])
# print(scorer(dataset))

# print(cfg['data'])
# formatter = FORMATTER_REGISTRY.get(cfg['data']['video-caption']['formatter'])(cfg['data']['video-caption'])
# dataset = formatter.load_dataset()
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
# index = random.randint(0, len(dataset))
# print(dataset[index]['captions'], type(dataset[index]['captions']))
# scorer = MODEL_REGISTRY.get('EMScorer')(cfg['video']['EMScorer'])
# print(f'the score of EMScorer is {scorer(dataset[index])}')
# for sample_batch in dataloader: 
#     print(type(sample_batch))
#     print(sample_batch)
#     print(f"the score of EMScorer is {scorer(sample_batch)}")
#     # for s in sample_batch:
#     #     print(f"the score of EMScorer is {scorer(s)}")
#     break
# for scorer_name, scorer_args in cfg['video'].items():
#     scorer = MODEL_REGISTRY.get(scorer_name)(scorer_args)
#     print(f'the score of {scorer_name} is {scorer(dataset[index])}')