import torch
import torch.nn as nn
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .head import VQAHead, IQAHead, VARHead


class BaseEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns



class DiViDeAddEvaluator(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys = 'fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4,4,4)}, fragments={"window_size": (4,4,4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size=="divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4,4,4), frag_biases=[0,0,0,0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b)   
        if divide_head:
            print(divide_head)
            for key in backbone:
                if key not in self.backbone_preserve_keys:
                    continue
                if var:
                    b = VARHead(**vqa_head)
                    print(b)
                else:
                    b = VQAHead(**vqa_head)
                print("Setting head:", key+"_head")
                setattr(self, key+"_head", b) 
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)
                print("init success")

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                
                scores = []
                feats = {}
                for key in vclips:
                    feat = getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                    if hasattr(self, key.split("_")[0]+"_head"):
                        scores += [getattr(self, key.split("_")[0]+"_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat.mean((-3,-2,-1))
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x,y:x+y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1,2,3,4))
            self.train()
            if return_pooled_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                if hasattr(self, key.split("_")[0]+"_head"):
                    scores += [getattr(self, key.split("_")[0]+"_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3,-2,-1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x,y:x+y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1,2,3,4))
                    print(scores.shape)
            
            if return_pooled_feats:
                return scores, feats
            return scores



class BaseImageEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        iqa_head=dict(),
    ):
        super().__init__()
        self.backbone = ImageBackbone(**backbone)
        self.iqa_head = IQAHead(**iqa_head)

    def forward(self, image, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(image)
                score = self.iqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(image)
            score = self.iqa_head(feat)
            return score

    def forward_with_attention(self, image):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(image, require_attn=True)
            score = self.iqa_head(feat)
            return score, avg_attns

if __name__ == "__main__":
    
    fusion_opt = {
        "anno_file": "./examplar_data_labels/KoNViD/labels.txt",
        "data_prefix": "../datasets/KoNViD",
        "sample_types": {"fragments": dict(fragments_h=4,fragments_w=4),
                         "resize": dict(size_h=128, size_w=128)},
        "phase": "train",
        "clip_len": 16,
        "frame_interval": 2,
        "num_clips": 1,
        "sampling_args": {}
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = FusionDataset(fusion_opt)
    
    model = DiViDeEvaluator({"resize":dict(window_size=(4,4,4)), 
                             "fragments":dict(window_size=(4,4,4))}).to(device)
    data = dataset[0]
    video = {}
    for key in fusion_opt["sample_types"]:
        video[key] = data[key].to(device).unsqueeze(0)
    print(torch.mean(model(video)))