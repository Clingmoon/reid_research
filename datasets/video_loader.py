from __future__ import print_function, absolute_import
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F_tf


class VideoTransform:
    """Video-specific transform that applies the SAME random parameters
    (flip, crop, erase) to all frames in a tracklet for temporal consistency."""

    def __init__(self, size_train, prob, padding, pixel_mean, pixel_std, re_prob):
        self.size_train = size_train
        self.prob = prob
        self.padding = padding
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.re_prob = re_prob

    def __call__(self, imgs):
        # imgs: list of PIL Images
        # 1. Resize
        imgs = [F_tf.resize(img, self.size_train, interpolation=3) for img in imgs]

        # 2. RandomHorizontalFlip (shared decision)
        if random.random() < self.prob:
            imgs = [F_tf.hflip(img) for img in imgs]

        # 3. Pad
        imgs = [F_tf.pad(img, self.padding) for img in imgs]

        # 4. RandomCrop (shared coordinates)
        i, j, h, w = T.RandomCrop.get_params(imgs[0], self.size_train)
        imgs = [F_tf.crop(img, i, j, h, w) for img in imgs]

        # 5. ToTensor
        imgs = [F_tf.to_tensor(img) for img in imgs]

        # 6. Normalize
        imgs = [F_tf.normalize(img, self.pixel_mean, self.pixel_std) for img in imgs]

        # 7. RandomErasing (shared region)
        if random.random() < self.re_prob:
            img0 = imgs[0]
            c, h, w = img0.shape
            area = h * w
            for _ in range(100):
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 1.0 / 0.3)
                eh = int(round(math.sqrt(target_area * aspect_ratio)))
                ew = int(round(math.sqrt(target_area / aspect_ratio)))
                if eh < h and ew < w:
                    ei = random.randint(0, h - eh)
                    ej = random.randint(0, w - ew)
                    v = torch.tensor(self.pixel_mean, dtype=img0.dtype).view(c, 1, 1)
                    imgs = [F_tf.erase(t, ei, ej, eh, ew, v, inplace=False) for t in imgs]
                    break

        return imgs


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'dense', 'rrs_train', 'rrs_test']

    def __init__(self, dataset, seq_len=8, sample='random', transform=None, video_transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.video_transform = video_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_single_item(index)

    def _get_single_item(self, index):
        img_paths, pid, camid, trackid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            indices = frame_indices[begin_index:end_index]
            while len(indices) < self.seq_len:
                indices.append(indices[-1])
            indices = np.array(indices)

            imgs = []
            for idx in indices:
                img = Image.open(img_paths[int(idx)]).convert('RGB')
                imgs.append(img)

            if self.video_transform is not None:
                imgs = self.video_transform(imgs)
            elif self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img_tensor = torch.stack(imgs, dim=0)  # (T, C, H, W)
            return img_tensor, pid, camid, trackid, ""

        elif self.sample == 'dense':
            cur_index = 0
            frame_indices = list(range(num))
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            last_seq = frame_indices[cur_index:]
            while len(last_seq) < self.seq_len:
                last_seq.append(last_seq[-1] if last_seq else 0)
            indices_list.append(last_seq)

            imgs_list = []
            for indices in indices_list:
                imgs = []
                for idx in indices:
                    img = Image.open(img_paths[int(idx)]).convert('RGB')
                    imgs.append(img)

                if self.video_transform is not None:
                    imgs = self.video_transform(imgs)
                elif self.transform is not None:
                    imgs = [self.transform(img) for img in imgs]

                imgs = torch.stack(imgs, 0)  # (T, C, H, W)
                imgs_list.append(imgs)

            imgs_tensor = torch.stack(imgs_list)  # (N, T, C, H, W)
            return imgs_tensor, pid, camid, trackid, ""

        elif self.sample == 'rrs_train':
            # Random sampling within strips
            S = self.seq_len
            frame_indices = list(range(num))
            if num < S:
                strip = list(range(num)) + [frame_indices[-1]] * (S - num)
                sample_clip = [[strip[s]] for s in range(S)]
            else:
                inter_val = math.ceil(num / S)
                strip = list(range(num)) + [frame_indices[-1]] * (inter_val * S - num)
                sample_clip = [strip[inter_val * s:inter_val * (s + 1)] for s in range(S)]

            sample_clip = np.array(sample_clip)
            idx = np.random.choice(sample_clip.shape[1], sample_clip.shape[0])
            number = sample_clip[np.arange(len(sample_clip)), idx]

            img_paths_arr = np.array(list(img_paths))
            imgs = [Image.open(img_paths_arr[n]).convert('RGB') for n in number]

            if self.video_transform is not None:
                imgs = self.video_transform(imgs)
            elif self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img_tensor = torch.stack(imgs, dim=0)  # (T, C, H, W)
            return img_tensor, pid, camid, trackid, ""

        elif self.sample == 'rrs_test':
            S = self.seq_len
            frame_indices = list(range(num))
            if num < S:
                strip = list(range(num)) + [frame_indices[-1]] * (S - num)
                sample_clip = [[strip[s]] for s in range(S)]
            else:
                inter_val = math.ceil(num / S)
                strip = list(range(num)) + [frame_indices[-1]] * (inter_val * S - num)
                sample_clip = [strip[inter_val * s:inter_val * (s + 1)] for s in range(S)]

            sample_clip = np.array(sample_clip)
            number = sample_clip[:, 0]

            img_paths_arr = np.array(list(img_paths))
            imgs = [Image.open(img_paths_arr[n]).convert('RGB') for n in number]

            if self.video_transform is not None:
                imgs = self.video_transform(imgs)
            elif self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img_tensor = torch.stack(imgs, dim=0)  # (T, C, H, W)
            return img_tensor, pid, camid, trackid, ""

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(
                self.sample, self.sample_methods))
