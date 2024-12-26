import torch

import argparse
import pickle as pkl

import decord
import numpy as np
import os

from dataflow.core import VideoScorer
from dataflow.Eval.video.dover.datasets.dover_datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dataflow.Eval.video.dover.models.evaluator import DOVER
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils import wget_model

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

@MODEL_REGISTRY.register()
class DOVERScorer(VideoScorer):

    def __init__(self, args_dict: dict, cfg=None):

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        super().__init__(args_dict)
        print(args_dict['model_path'], os.path.exists(args_dict['model_path']))
        if not os.path.exists(args_dict['model_path']):
            _ = wget_model('https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth', args_dict['model_path'])
            print(_)
        self.evaluator = DOVER(**args_dict["model_args"]).to("cuda")
        self.evaluator.load_state_dict(
            torch.load(args_dict["model_path"], map_location="cuda")
        )

        self.temporal_samplers = {}
        self.scorer_name = 'DOVERScorer'
        self.sample_args = args_dict["sample_types"]
        for stype, sopt in self.sample_args.items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
    
    # def evaluate(self, sample, rank=None):

    #     torch.manual_seed(42)
    #     torch.cuda.manual_seed(42)
    #     np.random.seed(42)
        
    #     views, _ = spatial_temporal_view_decomposition(
    #         sample['video'], self.sample_args, self.temporal_samplers
    #     )

    #     for k, v in views.items():
    #         num_clips = self.sample_args[k].get("num_clips", 1)
    #         views[k] = (
    #             ((v.permute(1, 2, 3, 0) - mean) / std)
    #             .permute(3, 0, 1, 2)
    #             .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
    #             .transpose(0, 1)
    #             .to('cuda')
    #         )
    #         print(views[k].shape)
        
    #     print(views.keys())
    #     results = [ r.mean().item() for r in self.evaluator(views) ]

    #     return results
    def init_score(self, len_dataset):
        '''
        return empty score dict for this scorer
        eg: {'Default': np.array([-1] * len_dataset)}
        '''
        return {'technical': np.array([np.nan] * len_dataset), 'aesthetic': np.array([np.nan] * len_dataset)}

    def evaluate_batch(self, sample_list: list, rank=None):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        frag_list = {}
        num_video = len(sample_list['video'])
        for sample in sample_list['video']:
            views, _ = spatial_temporal_view_decomposition(
                sample, self.sample_args, self.temporal_samplers
            )

            for k, v in views.items():
                num_clips = self.sample_args[k].get("num_clips", 1)
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to('cuda')
                )
                if k not in frag_list.keys():
                    frag_list[k] = views[k]
                else:
                    frag_list[k] = torch.cat((frag_list[k], views[k]), dim=0)
            print(views.keys())


        results = self.evaluator(frag_list)
        tech_res = results[0].view(min(self.batch_size, num_video), results[0].shape[0] // min(self.batch_size, num_video), *results[0].shape[1:]).mean(dim=(1,2,3,4,5)).cpu()
        aes_res = results[1].view(min(self.batch_size, num_video), results[1].shape[0] // min(self.batch_size, num_video), *results[1].shape[1:]).mean(dim=(1,2,3,4,5)).cpu()
        
        return {list(views.keys())[0]: tech_res, list(views.keys())[1]: aes_res}

    # def __call__(self, sample, rank=None):
    #     if isinstance(sample, list):
    #         return self.evaluate_batch(sample, rank)
    #     return self.evaluate(sample, rank)