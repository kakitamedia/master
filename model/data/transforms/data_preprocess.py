from copy import deepcopy
import time
import cv2
import random

import torch

from .transforms import *

class TrainTransform:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            RandomResize(),
            RandomMirror(),
            PhotometricDistort(),
            Normalize(cfg),
            # ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.augment(image, boxes, labels)
        return image, boxes, labels

class AETrainTransform:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            RandomResize(),
            RandomMirror(),
            PhotometricDistort(),
            Normalize(cfg),
        ])

        self.crops = [
            RandomCrop(crop_size=cfg.TEST.AE_INPUT_PATCH*down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS
        ]

        self.valid_scale = cfg.MODEL.VALID_SCALE
        self.to_tensor = ToTensor()

    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.augment(image, boxes, labels)

        images = [0 for _ in range(len(self.valid_scale))]
        for i in self.valid_scale:
            img, _, _ = self.crops[i](deepcopy(image))
            img, _, _ = self.to_tensor(img)
            images[i] = img

        return images, None, None


class EvalTransform:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            ConstantPadding(),
            Normalize(cfg),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.augment(image, boxes, labels)
        images = [image for _ in range(3)]
        # images = [image]
        return images, boxes, labels


class TrainTargetTransform:
    def __init__(self, cfg):
        scale_ranges = cfg.DATASETS.SCALE_RANGES
        max_shifts = cfg.DATASETS.MAX_SHIFTS
        crop_sizes = cfg.DATASETS.CROP_SIZES
        down_ratios = cfg.MODEL.DOWN_RATIOS
        self.valid_scale = cfg.MODEL.VALID_SCALE
        self.sample_crops = [
            SampleCrop(cfg, scale_range=scale_range, max_shift=max_shift, crop_size=crop_size)
            for scale_range, max_shift, crop_size in zip(scale_ranges, max_shifts, crop_sizes)
        ]
        self.make_heatmaps = [
            MakeHeatmap(cfg, down_ratio=down_ratio)
            for down_ratio in down_ratios
        ]
        self.to_tensor = ToTensor()


    def __call__(self, image, boxes, labels):
        images, targets = [0 for _ in range(len(self.valid_scale))], [0 for _ in range(len(self.valid_scale))]
        # for crop, make_heatmap in zip(self.sample_crops, self.make_heatmaps):
        # start = time.time()
        for i in self.valid_scale:
            img, box, lab = self.sample_crops[i](deepcopy(image), deepcopy(boxes), deepcopy(labels))
            img, tar = self.make_heatmaps[i](img, box, lab)
            
            img, _, _ = self.to_tensor(img)
            for k in tar.keys():
                tar[k] = torch.from_numpy(tar[k])
            images[i] = img
            targets[i] = tar
        # end = time.time()
        # print(end - start)

        return images, targets

class TrainTragetTransform_ContentsMatch:
    def __init__(self, cfg):
        scale_ranges = cfg.DATASETS.SCALE_RANGES
        max_shifts = cfg.DATASETS.MAX_SHIFTS
        self.crop_sizes = cfg.DATASETS.CROP_SIZES
        down_ratios = cfg.MODEL.DOWN_RATIOS

        self.valid_scale = cfg.MODEL.VALID_SCALE

        self.sample_crop = SampleCrop(cfg, scale_range=scale_ranges[0], max_shift=max_shifts[0], crop_size=self.crop_sizes[0])
        self.make_heatmap = MakeHeatmap(cfg, down_ratio=down_ratios[0])
        self.to_tensor = ToTensor()

        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]


    def __call__(self, image, boxes, labels):
        images, targets = [0 for _ in range(len(self.valid_scale))], [0 for _ in range(len(self.valid_scale))]
        
        img, box, lab = self.sample_crop(image, boxes, labels)
        img, tar = self.make_heatmap(img, box, lab)

        interp_method = self.interp_methods[random.randint(0, 4)]
        for i in self.valid_scale:
            temp, _, _ = self.to_tensor(cv2.resize(deepcopy(img), (self.crop_sizes[i], self.crop_sizes[i]), interpolation=interp_method))
            images[i] = temp
            targets[i] = tar
        
        return images, targets


# class TrainTargetTransform:
#     def __init__(self, cfg):
#         scale_ranges = cfg.DATASETS.SCALE_RANGES
#         max_shifts = cfg.DATASETS.MAX_SHIFTS
#         crop_sizes = cfg.DATASETS.CROP_SIZES
#         down_ratios = cfg.MODEL.DOWN_RATIOS
#         self.valid_scale = cfg.MODEL.VALID_SCALE
#         self.augments = [
#             SampleCrop_MakeHeatmap(cfg, scale_range=scale_range, max_shift=max_shift, crop_size=crop_size, down_ratio=down_ratio)
#             for scale_range, max_shift, crop_size, down_ratio in zip(scale_ranges, max_shifts, crop_sizes, down_ratios)
#         ]
#         self.to_tensor = ToTensor()


#     def __call__(self, image, boxes, labels):
#         images, targets = [0 for _ in range(4)], [0 for _ in range(4)]
#         for i in self.valid_scale:
#             img, tar = self.augments[i](deepcopy(image), deepcopy(boxes), deepcopy(labels))
            
#             img, _, _ = self.to_tensor(img)
#             for k in tar.keys():
#                 tar[k] = torch.from_numpy(tar[k])

#             images[i] = img
#             targets[i] = tar


#         return images, targets