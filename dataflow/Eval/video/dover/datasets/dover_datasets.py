import copy
import glob
import os
import os.path as osp
import random
from functools import lru_cache

import cv2
import decord
import numpy as np
import skvideo.io
import torch
import torchvision
from decord import VideoReader, cpu, gpu
from tqdm import tqdm

random.seed(42)

decord.bridge.set_bridge("torch")


def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    upsample=-1,
    **kwargs,
):
    if upsample > 0:
        old_h, old_w = video.shape[-2], video.shape[-1]
        if old_h >= old_w:
            w = upsample
            h = int(upsample * old_h / old_w)
        else:
            h = upsample
            w = int(upsample * old_w / old_h)
        
        video = get_resized_video(video, h, w)
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:

        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_upsample:

        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    # target_videos.append(video[:,t_s:t_e,h_so:h_eo,w_so:w_eo])
    # target_video = torch.stack(target_videos, 0).reshape((dur_t // aligned, fragments, fragments,) + target_videos[0].shape).permute(3,0,4,1,5,2,6)
    # target_video = target_video.reshape((-1, dur_t,) + size) ## Splicing Fragments
    return target_video


@lru_cache
def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))


def get_resized_video(
    video, size_h=224, size_w=224, random_crop=False, arp=False, **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video


def get_arp_resized_video(
    video, short_edge=224, train=False, **kwargs,
):
    if train:  ## if during training, will random crop into square and then resize
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h : rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h : rnd_h + ori_short_edge]
    ori_short_edge = min(video.shape[-2:])
    scale_factor = short_edge / ori_short_edge
    ovideo = video
    video = torch.nn.functional.interpolate(
        video / 255.0, scale_factors=scale_factor, mode="bilinear"
    )
    video = (video * 255.0).type_as(ovideo)
    return video


def get_arp_fragment_video(
    video, short_fragments=7, fsize=32, train=False, **kwargs,
):
    if (
        train
    ):  ## if during training, will random crop into square and then get fragments
        res_h, res_w = video.shape[-2:]
        ori_short_edge = min(video.shape[-2:])
        if res_h > ori_short_edge:
            rnd_h = random.randrange(res_h - ori_short_edge)
            video = video[..., rnd_h : rnd_h + ori_short_edge, :]
        elif res_w > ori_short_edge:
            rnd_w = random.randrange(res_w - ori_short_edge)
            video = video[..., :, rnd_h : rnd_h + ori_short_edge]
    kwargs["fsize_h"], kwargs["fsize_w"] = fsize, fsize
    res_h, res_w = video.shape[-2:]
    if res_h > res_w:
        kwargs["fragments_w"] = short_fragments
        kwargs["fragments_h"] = int(short_fragments * res_h / res_w)
    else:
        kwargs["fragments_h"] = short_fragments
        kwargs["fragments_w"] = int(short_fragments * res_w / res_h)
    return get_spatial_fragments(video, **kwargs)


def get_cropped_video(
    video, size_h=224, size_w=224, **kwargs,
):
    kwargs["fragments_h"], kwargs["fragments_w"] = 1, 1
    kwargs["fsize_h"], kwargs["fsize_w"] = size_h, size_w
    return get_spatial_fragments(video, **kwargs)


def get_single_view(
    video, sample_type="aesthetic", **kwargs,
):
    if sample_type.startswith("aesthetic"):
        video = get_resized_video(video, **kwargs)
    elif sample_type.startswith("technical"):
        video = get_spatial_fragments(video, **kwargs)
    elif sample_type == "original":
        return video

    return video


def spatial_temporal_view_decomposition(
    video_path, sample_types, samplers, is_train=False, augment=False,
):
    video = {}
    if video_path.endswith(".yuv"):
        print("This part will be deprecated due to large memory cost.")
        ## This is only an adaptation to LIVE-Qualcomm
        ovideo = skvideo.io.vread(
            video_path, 1080, 1920, inputdict={"-pix_fmt": "yuvj420p"}
        )
        for stype in samplers:
            frame_inds = samplers[stype](ovideo.shape[0], is_train)
            imgs = [torch.from_numpy(ovideo[idx]) for idx in frame_inds]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        del ovideo
    else:
        decord.bridge.set_bridge("torch")
        vreader = VideoReader(video_path)
        ### Avoid duplicated video decoding!!! Important!!!!
        all_frame_inds = []
        frame_inds = {}
        for stype in samplers:
            frame_inds[stype] = samplers[stype](len(vreader), is_train)
            all_frame_inds.append(frame_inds[stype])

        ### Each frame is only decoded one time!!!
        all_frame_inds = np.concatenate(all_frame_inds, 0)
        frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

        for stype in samplers:
            imgs = [frame_dict[idx] for idx in frame_inds[stype]]
            video[stype] = torch.stack(imgs, 0).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)
    return sampled_video, frame_inds


import random

import numpy as np


class UnifiedFrameSampler:
    def __init__(
        self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0.0,
    ):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):

        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )

        drop = random.sample(
            list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate)
        )
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)


class ViewDecompositionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling

        super().__init__()

        self.weight = opt.get("weight", 0.5)
        
        self.fully_supervised = opt.get("fully_supervised", False)
        print("Fully supervised:", self.fully_supervised)
        
        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)
        if self.data_backend == "petrel":
            from petrel_client import client

            self.client = client.Client(enable_mc=True)

        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}
        for stype, sopt in opt["sample_types"].items():
            if "t_frag" not in sopt:
                # resized temporal sampling for TQE in DOVER
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
            print(
                stype + " branch sampled frames:",
                self.samplers[stype](240, self.phase == "train"),
            )

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            try:
                with open(self.ann_file, "r") as fin:
                    for line in fin:
                        line_split = line.strip().split(",")
                        filename, a, t, label = line_split
                        if self.fully_supervised:
                            label = float(a), float(t), float(label)
                        else:
                            label = float(label)
                        filename = osp.join(self.data_prefix, filename)
                        self.video_infos.append(dict(filename=filename, label=label))
            except:
                #### No Label Testing
                video_filenames = []
                for (root, dirs, files) in os.walk(self.data_prefix, topdown=True):
                    for file in files:
                        if file.endswith(".mp4"):
                            video_filenames += [os.path.join(root, file)]
                print(len(video_filenames))
                video_filenames = sorted(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]

        try:
            ## Read Original Frames
            ## Process Frames
            data, frame_inds = spatial_temporal_view_decomposition(
                filename,
                self.sample_types,
                self.samplers,
                self.phase == "train",
                self.augment and (self.phase == "train"),
            )

            for k, v in data.items():
                data[k] = ((v.permute(1, 2, 3, 0) - self.mean) / self.std).permute(
                    3, 0, 1, 2
                )

            data["num_clips"] = {}
            for stype, sopt in self.sample_types.items():
                data["num_clips"][stype] = sopt["num_clips"]
            data["frame_inds"] = frame_inds
            data["gt_label"] = label
            data["name"] = filename  # osp.basename(video_info["filename"])
        except:
            # exception flow
            return {"name": filename}

        return data

    def __len__(self):
        return len(self.video_infos)
