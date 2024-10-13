import time
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d

from .conv_backbone import convnext_3d_small, convnext_3d_tiny, convnextv2_3d_pico, convnextv2_3d_femto
from .head import IQAHead, VARHead, VQAHead
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny


class BaseEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), vqa_head=dict(),
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


class DOVER(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(
            resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
        ),
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
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "swin_tiny":
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == "conv_tiny":
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == "conv_small":
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == "conv_femto":
                b = convnextv2_3d_femto(pretrained=True)
            elif t_backbone_size == "conv_pico":
                b = convnextv2_3d_pico(pretrained=True)
            elif t_backbone_size == "xclip":
                raise NotImplementedError
                # b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
        if divide_head:
            for key in backbone:
                pre_pool = False #if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                b = VQAHead(pre_pool=pre_pool, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)

    def forward(
        self,
        vclips,
        inference=True,
        return_pooled_feats=False,
        return_raw_feats=False,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in vclips:
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                        vclips[key], multi=self.multi, layer=self.layer, **kwargs
                    )
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat
                    if return_raw_feats:
                        feats[key] = feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            if return_pooled_feats or return_raw_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
    def forward_head(
        self,
        feats,
        inference=True,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in feats:
                    feat = feats[key]
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
class MinimumDOVER(nn.Module):
    def __init__(self):
        super().__init__()
        self.technical_backbone = VideoBackbone()
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=True)
        self.technical_head = VQAHead(pre_pool=False, in_channels=768)
        self.aesthetic_head = VQAHead(pre_pool=False, in_channels=768)


    def forward(self,aesthetic_view, technical_view):
        self.eval()
        with torch.no_grad():
            aesthetic_score = self.aesthetic_head(self.aesthetic_backbone(aesthetic_view))
            technical_score = self.technical_head(self.technical_backbone(technical_view))
            
        aesthetic_score_pooled = torch.mean(aesthetic_score, (1,2,3,4))
        technical_score_pooled = torch.mean(technical_score, (1,2,3,4))
        return [aesthetic_score_pooled, technical_score_pooled]



class BaseImageEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), iqa_head=dict(),
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
