import os
import torch
import decord
from dataflow.Eval.video.fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from dataflow.Eval.video.fastvqa.models import DiViDeAddEvaluator
import numpy as np
import jsonargparse
from dataflow.core import Scorer, VideoScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils import wget_model

def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml", 
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml", 
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml", 
    "FAST-VQA": "./options/fast/fast-b.yml", 
    "FAST-VQA-M": "./options/fast/fast-m.yml", 
}

@MODEL_REGISTRY.register()
class FastVQAScorer(VideoScorer):

    def __init__(self, args_dict: dict, cfg=None):
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        super().__init__(args_dict)
        self.scorer_name = 'FastVQAScorer'
        self.sample_args = args_dict['fragments']
        if self.sample_args.get('t_frag', 1) > 1:
            self.sampler = FragmentSampleFrames(fsize_t=self.sample_args["clip_len"] // self.sample_args.get("t_frag",1),
                                           fragments_t=self.sample_args.get("t_frag",1),
                                           num_clips=self.sample_args.get("num_clips",1),
                                          )
        else:
            self.sampler = SampleFrames(clip_len = self.sample_args["clip_len"], num_clips = self.sample_args["num_clips"])
        if not os.path.exists(args_dict['model_path']):
            wget_model('https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth', args_dict['model_path'])

        self.evaluator = DiViDeAddEvaluator(**args_dict['model_args']).to('cuda')
        self.evaluator.load_state_dict(torch.load(args_dict["model_path"], map_location='cuda')["state_dict"])

    # def evaluate(self, sample: dict, rank=None):
    #     torch.manual_seed(42)
    #     torch.cuda.manual_seed(42)
    #     np.random.seed(42)
        
    #     vsamples = {}
    #     video_reader = decord.VideoReader(sample['video'])
    #     num_clips = self.sample_args.get("num_clips",1)
    #     frames = self.sampler(len(video_reader))
    #     print("Sampled frames are", frames)
    #     frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
    #     imgs = [frame_dict[idx] for idx in frames]
    #     video = torch.stack(imgs, 0)
    #     video = video.permute(3, 0, 1, 2)

    #     ## Sample Spatially
    #     sampled_video = get_spatial_fragments(video, **self.sample_args)
    #     mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
    #     sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        
    #     sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
    #     vsamples['fragments'] = sampled_video.to('cuda')
    #     print(sampled_video.shape)
    #     result = self.evaluator(vsamples)
    #     score = sigmoid_rescale(result.mean().item(), model='FAST-VQA')
    #     print(f"The quality score of the video (range [0,1]) is {score:.5f}.")

    #     return score

    def evaluate_batch(self, sample_list, rank=None):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        vsamples = {}
        frag_list = None
        num_video = len(sample_list['video'])
        for sample in sample_list['video']:
            video_reader = decord.VideoReader(sample)
            num_clips = self.sample_args.get("num_clips",1)
            frames = self.sampler(len(video_reader))
            print("Sampled frames are", frames)
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)

            ## Sample Spatially
            sampled_video = get_spatial_fragments(video, **self.sample_args)
            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
            print(sampled_video.shape)
            if frag_list is None:
                frag_list = sampled_video
            else:
                frag_list = torch.cat((frag_list, sampled_video), dim=0)
        vsamples['fragments'] = frag_list.to('cuda')
        print(vsamples['fragments'].shape)
        result = self.evaluator(vsamples)
        reshaped_result = result.view(min(self.batch_size, num_video), result.shape[0] // min(self.batch_size, num_video), *result.shape[1:])
        print(result.shape, reshaped_result.shape)
        score = sigmoid_rescale(reshaped_result.mean(dim=(1,2,3,4,5)).cpu(), model='FAST-VQA')
        print(f"The quality score of the video (range [0,1]) is {score.numpy()}.")

        return score

@MODEL_REGISTRY.register()
class FasterVQAScorer(VideoScorer):
    def __init__(self, args_dict: dict, cfg=None):
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        print(args_dict)
        super().__init__(args_dict)
        self.scorer_name = 'FasterVQAScorer'
        self.sample_args = args_dict['fragments']
        if self.sample_args.get('t_frag', 1) > 1:
            self.sampler = FragmentSampleFrames(fsize_t=self.sample_args["clip_len"] // self.sample_args.get("t_frag",1),
                                           fragments_t=self.sample_args.get("t_frag",1),
                                           num_clips=self.sample_args.get("num_clips",1),
                                          )
        else:
            self.sampler = SampleFrames(clip_len = self.sample_args["clip_len"], num_clips = self.sample_args["num_clips"])
        print(args_dict['model_path'], os.path.exists(args_dict['model_path']))
        if not os.path.exists(args_dict['model_path']):
            wget_model('https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST_VQA_3D_1_1.pth', args_dict['model_path'])
        self.evaluator = DiViDeAddEvaluator(**args_dict['model_args']).to('cuda')
        self.evaluator.load_state_dict(torch.load(args_dict["model_path"], map_location='cuda')["state_dict"])


    # def evaluate(self, sample, rank=None):

    #     torch.manual_seed(42)
    #     torch.cuda.manual_seed(42)    
    #     np.random.seed(42)
        
    #     vsamples = {}
    #     video_reader = decord.VideoReader(sample['video'])
    #     num_clips = self.sample_args.get("num_clips",1)
    #     frames = self.sampler(len(video_reader))
    #     print("Sampled frames are", frames)
    #     frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
    #     imgs = [frame_dict[idx] for idx in frames]
    #     video = torch.stack(imgs, 0)
    #     video = video.permute(3, 0, 1, 2)

    #     ## Sample Spatially
    #     sampled_video = get_spatial_fragments(video, **self.sample_args)
    #     mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
    #     sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        
    #     sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
    #     vsamples['fragments'] = sampled_video.to('cuda')
    #     print(sampled_video.shape)
    #     result = self.evaluator(vsamples)
    #     score = sigmoid_rescale(result.mean().item(), model='FasterVQA')
    #     print(f"The quality score of the video (range [0,1]) is {score:.5f}.")
        
    #     return score

    def evaluate_batch(self, sample_list):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        vsamples = {}
        frag_list = None
        num_video = len(sample_list['video'])
        for sample in sample_list['video']:
            video_reader = decord.VideoReader(sample)
            num_clips = self.sample_args.get("num_clips",1)
            frames = self.sampler(len(video_reader))
            print("Sampled frames are", frames)
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)

            ## Sample Spatially
            sampled_video = get_spatial_fragments(video, **self.sample_args)
            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
            if frag_list is None:
                frag_list = sampled_video
            else:
                frag_list = torch.cat((frag_list, sampled_video), dim=0)
        vsamples['fragments'] = frag_list.to('cuda')
        print(vsamples['fragments'].shape)
        result = self.evaluator(vsamples)
        print(result.shape)
        reshaped_result = result.view(min(self.batch_size, num_video), result.shape[0] // min(self.batch_size, num_video), *result.shape[1:])
        print(reshaped_result.shape)
        score = sigmoid_rescale(reshaped_result.mean(dim=(1,2,3,4,5)).cpu().detach(), model='FasterVQA')
        # print(f"The quality score of the video (range [0,1]) is {score:.5f}.")
        
        return score



    # def __call__(self, sample, rank=None):
    #     if isinstance(sample, list):
    #         return self.evaluate_batch(sample, rank)
    #     return self.evaluate(sample, rank)

