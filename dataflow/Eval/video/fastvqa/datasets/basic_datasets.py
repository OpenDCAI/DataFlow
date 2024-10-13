import decord
from decord import VideoReader
from decord import cpu, gpu
import os.path as osp
import numpy as np
import torch, torchvision
from tqdm import tqdm
import cv2
import skvideo.io

import random

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
    fallback_type="upsample",
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

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


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def get_frame_indices(self, num_frames):

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
        return np.concatenate(ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []
        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]
        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds


class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


class FastVQAPlusPlusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        frame_interval=2,
        aligned=32,
        fragments=(8, 8, 8),
        fsize=(4, 32, 32),
        num_clips=1,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
        fallback_type="oversample",
    ):
        """
        Fragments.
        args:
            fragments: G_f as in the paper.
            fsize: S_f as in the paper.
            nfrags: number of samples (spatially) as in the paper.
            num_clips: number of samples (temporally) as in the paper.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.clip_len = fragments[0] * fsize[0]
        self.aligned = aligned
        self.fallback_type = fallback_type
        self.sampler = FragmentSampleFrames(
            fsize[0], fragments[0], frame_interval, num_clips
        )
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self,
        index,
        tocache=False,
        need_original_frames=False,
    ):
        if tocache or self.cache is None:
            fx, fy = self.fragments[1:]
            fsx, fsy = self.fsize[1:]
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            if filename.endswith(".yuv"):
                video = skvideo.io.vread(filename, 1080, 1920, inputdict={'-pix_fmt':'yuvj420p'})
                frame_inds = self.sampler(video.shape[0], self.phase == "train")
                imgs = [torch.from_numpy(video[idx]) for idx in frame_inds]
            else:
                vreader = VideoReader(filename)
                frame_inds = self.sampler(len(vreader), self.phase == "train")
                frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
                imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            if self.nfrags == 1:
                vfrag = get_spatial_fragments(
                    video,
                    fx,
                    fy,
                    fsx,
                    fsy,
                    aligned=self.aligned,
                    fallback_type=self.fallback_type,
                )
            else:
                vfrag = get_spatial_fragments(
                    video,
                    fx,
                    fy,
                    fsx,
                    fsy,
                    aligned=self.aligned,
                    fallback_type=self.fallback_type,
                )
                for i in range(1, self.nfrags):
                    vfrag = torch.cat(
                        (
                            vfrag,
                            get_spatial_fragments(
                                video,
                                fragments,
                                fx,
                                fy,
                                fsx,
                                fsy,
                                aligned=self.aligned,
                                fallback_type=self.fallback_type,
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class FragmentVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        fragments=7,
        fsize=32,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Fragments.
        args:
            fragments: G_f as in the paper.
            fsize: S_f as in the paper.
            nfrags: number of samples as in the paper.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.aligned = aligned
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self,
        index,
        fragments=-1,
        fsize=-1,
        tocache=False,
        need_original_frames=False,
    ):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            if filename.endswith(".yuv"):
                video = skvideo.io.vread(filename, 1080, 1920, inputdict={'-pix_fmt':'yuvj420p'})
                frame_inds = self.sampler(video.shape[0], self.phase == "train")
                imgs = [torch.from_numpy(video[idx]) for idx in frame_inds]
            else:
                vreader = VideoReader(filename)
                frame_inds = self.sampler(len(vreader), self.phase == "train")
                frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
                imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            if self.nfrags == 1:
                vfrag = get_spatial_fragments(
                    video, fragments, fragments, fsize, fsize, aligned=self.aligned
                )
            else:
                vfrag = get_spatial_fragments(
                    video, fragments, fragments, fsize, fsize, aligned=self.aligned
                )
                for i in range(1, self.nfrags):
                    vfrag = torch.cat(
                        (
                            vfrag,
                            get_spatial_fragments(
                                video,
                                fragments,
                                fragments,
                                fsize,
                                fsize,
                                aligned=self.aligned,
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class ResizedVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        size=224,
        cache_in_memory=False,
        phase="test",
    ):
        """
        Using resizing.
        """
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.size = size
        self.aligned = aligned
        self.sampler = SampleFrames(clip_len, frame_interval, num_clips)
        self.video_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching resized videos"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(self, index, tocache=False, need_original_frames=False):
        if tocache or self.cache is None:
            video_info = self.video_infos[index]
            filename = video_info["filename"]
            label = video_info["label"]
            vreader = VideoReader(filename)
            frame_inds = self.sampler(len(vreader), self.phase == "train")
            frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
            imgs = [frame_dict[idx] for idx in frame_inds]
            img_shape = imgs[0].shape
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            video = torch.nn.functional.interpolate(video, size=(self.size, self.size))
            if tocache:
                return (vfrag, frame_inds, label, img_shape)
        else:
            vfrag, frame_inds, label, img_shape = self.cache[index]
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)
        data = {
            "video": vfrag.reshape(
                (-1, self.num_clips, self.clip_len) + vfrag.shape[2:]
            ).transpose(
                0, 1
            ),  # B, V, T, C, H, W
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_video"] = video.reshape(
                (-1, self.nfrags * self.num_clips, self.clip_len) + video.shape[2:]
            ).transpose(0, 1)
        return data

    def __len__(self):
        return len(self.video_infos)


class CroppedVideoDataset(FragmentVideoDataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        aligned=32,
        size=224,
        ncrops=1,
        cache_in_memory=False,
        phase="test",
    ):

        """
        Regard Cropping as a special case for Fragments in Grid 1*1.
        """
        super().__init__(
            ann_file,
            data_prefix,
            clip_len=clip_len,
            frame_interval=frame_interval,
            num_clips=num_clips,
            aligned=aligned,
            fragments=1,
            fsize=224,
            nfrags=ncrops,
            cache_in_memory=cache_in_memory,
            phase=phase,
        )


class FragmentImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        fragments=7,
        fsize=32,
        nfrags=1,
        cache_in_memory=False,
        phase="test",
    ):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.fragments = fragments
        self.fsize = fsize
        self.nfrags = nfrags
        self.image_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.image_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.image_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self, index, fragments=-1, fsize=-1, tocache=False, need_original_frames=False
    ):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            image_info = self.image_infos[index]
            filename = image_info["filename"]
            label = image_info["label"]
            try:
                img = torchvision.io.read_image(filename)
            except:
                img = cv2.imread(filename)
                img = torch.from_numpy(img[:, :, [2, 1, 0]]).permute(2, 0, 1)
            img_shape = img.shape[1:]
            image = img.unsqueeze(1)
            if self.nfrags == 1:
                ifrag = get_spatial_fragments(image, fragments, fragments, fsize, fsize)
            else:
                ifrag = get_spatial_fragments(image, fragments, fragments, fsize, fsize)
                for i in range(1, self.nfrags):
                    ifrag = torch.cat(
                        (
                            ifrag,
                            get_spatial_fragments(
                                image, fragments, fragments, fsize, fsize
                            ),
                        ),
                        1,
                    )
            if tocache:
                return (ifrag, label, img_shape)
        else:
            ifrag, label, img_shape = self.cache[index]
        if self.nfrags == 1:
            ifrag = (
                ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
                .squeeze(0)
                .permute(2, 0, 1)
            )
        else:
            ### During testing, one image as a batch
            ifrag = (
                ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
                .squeeze(0)
                .permute(0, 3, 1, 2)
            )
        data = {
            "image": ifrag,
            "gt_label": label,
            "original_shape": img_shape,
            "name": filename,
        }
        if need_original_frames:
            data["original_image"] = image.squeeze(1)
        return data

    def __len__(self):
        return len(self.image_infos)


class ResizedImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        size=224,
        cache_in_memory=False,
        phase="test",
    ):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.size = size
        self.image_infos = []
        self.phase = phase
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        if isinstance(self.ann_file, list):
            self.image_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split(",")
                    filename, _, _, label = line_split
                    label = float(label)
                    filename = osp.join(self.data_prefix, filename)
                    self.image_infos.append(dict(filename=filename, label=label))
        if cache_in_memory:
            self.cache = {}
            for i in tqdm(range(len(self)), desc="Caching fragments"):
                self.cache[i] = self.__getitem__(i, tocache=True)
        else:
            self.cache = None

    def __getitem__(
        self, index, fragments=-1, fsize=-1, tocache=False, need_original_frames=False
    ):
        if tocache or self.cache is None:
            if fragments == -1:
                fragments = self.fragments
            if fsize == -1:
                fsize = self.fsize
            image_info = self.image_infos[index]
            filename = image_info["filename"]
            label = image_info["label"]
            img = torchvision.io.read_image(filename)
            img_shape = img.shape[1:]
            image = img.unsqueeze(1)
            if self.nfrags == 1:
                ifrag = get_spatial_fragments(image, fragments, fsize)
            else:
                ifrag = get_spatial_fragments(image, fragments, fsize)
                for i in range(1, self.nfrags):
                    ifrag = torch.cat(
                        (ifrag, get_spatial_fragments(image, fragments, fsize)), 1
                    )
            if tocache:
                return (ifrag, label, img_shape)
        else:
            ifrag, label, img_shape = self.cache[index]
        ifrag = (
            ((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std)
            .squeeze(0)
            .permute(2, 0, 1)
        )
        data = {
            "image": ifrag,
            "gt_label": label,
            "original_shape": img_shape,
        }
        if need_original_frames:
            data["original_image"] = image.squeeze(1)
        return data

    def __len__(self):
        return len(self.image_infos)


class CroppedImageDataset(FragmentImageDataset):
    def __init__(
        self,
        ann_file,
        data_prefix,
        size=224,
        ncrops=1,
        cache_in_memory=False,
        phase="test",
    ):

        """
        Regard Cropping as a special case for Fragments in Grid 1*1.
        """
        super().__init__(
            ann_file,
            data_prefix,
            fragments=1,
            fsize=224,
            nfrags=ncrops,
            cache_in_memory=cache_in_memory,
            phase=phase,
        )
