# -*- coding: utf-8 -*-
from __future__ import print_function, division

import json
import random
from typing import Tuple, Union
from .abstract_transform import AbstractTransform
from .image_process import *
from .ctr_transform import OffsetPlusCTR

class RandomCrop(AbstractTransform):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """
    def __init__(self,
                 p: float = 0.5,
                 crop_ratio: Union[Tuple[float], float] = 0.9, 
                 ctr_margin: Union[Tuple[int], int] = 10,
                 crop_z: bool = True,
                 crop_y: bool = True,
                 crop_x: bool = True,
                 pad_value: int = 0):
        self.p = p
        self.pad_value = pad_value
        self.crop_z = crop_z
        self.crop_y = crop_y
        self.crop_x = crop_x
        if isinstance(crop_ratio, (int, float)):
            assert crop_ratio > 0 and crop_ratio <= 1
            self.crop_ratio = (crop_ratio, crop_ratio, crop_ratio)
        else:
            self.crop_ratio = crop_ratio
        
        if isinstance(ctr_margin, (int, float)):
            assert ctr_margin >= 0
            self.ctr_margin = (ctr_margin, ctr_margin, ctr_margin)
        else:
            self.ctr_margin = ctr_margin

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape # [C, D, H, W]
        channels, depth, height, width = image.shape
        input_shape = np.array([depth, height, width], dtype=np.int32)
        ratio_d, ratio_h, ratio_w = self.crop_ratio
        # input_dim = len(input_shape) - 1

        # Decide the crop size
        if self.crop_z and random.random() < self.p:
            crop_depth = (random.random() * (1 - ratio_d) + ratio_d) * depth
        else:
            crop_depth = depth
            
        if self.crop_y and random.random() < self.p:
            crop_height = (random.random() * (1 - ratio_h) + ratio_h) * height
        else:
            crop_height = height
            
        if self.crop_x and random.random() < self.p:
            crop_width = (random.random() * (1 - ratio_w) + ratio_w) * width
        else:
            crop_width = width
            
        crop_shape = np.array([crop_depth, crop_height, crop_width]).astype(np.int32)
        crop_shape = np.clip(crop_shape, a_min=0, a_max=input_shape)
        if np.all(crop_shape == input_shape):
            return sample
        
        # Decide the crop position
        max_crop_pos = input_shape - crop_shape
        
        bb_min = np.random.randint(0, max_crop_pos + 1)
        bb_min = np.clip(bb_min, a_min=0, a_max=None)
        bb_max = bb_min + crop_shape
        bb_max = np.clip(bb_max, a_min=None, a_max=input_shape)
        crop_shape = bb_max - bb_min
        # Check if the ctr is within the crop region
        if 'ctr' in sample and len(sample['ctr']) > 0:
            ctr = sample['ctr']
            rad = sample['rad']
            ctr_min = ctr - rad / 2 - self.ctr_margin
            ctr_max = ctr + rad / 2 + self.ctr_margin
            ctr_min = np.maximum(ctr_min, 0)
            ctr_max = np.minimum(ctr_max, input_shape)
            
            # If the ctr is not within the crop region, then we do not crop the image
            if (not np.all(ctr_min > bb_min)) or (not np.all(ctr_max < bb_max)):
                return sample
        # Crop the image
        crop_image = np.ones_like(image) * self.pad_value
        crop_min = ((input_shape - crop_shape) / 2).astype(np.int32)
        crop_max = crop_min + crop_shape
        crop_image[:, crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]] = \
            image[:, bb_min[0]:bb_max[0], bb_min[1]:bb_max[1], bb_min[2]:bb_max[2]].copy()
        sample['image'] = crop_image
        # Update the center
        if 'ctr' in sample and len(sample['ctr']) > 0:
            sample['ctr'] = sample['ctr'].copy() - bb_min[np.newaxis, :] + crop_min[np.newaxis, :]
            sample['ctr_transform'].append(OffsetPlusCTR(-bb_min + crop_min))
            
        ##TODO: add feature transform
            
        return sample

class RandomMaskCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])
    """

    def __init__(self, output_size, pos_ratio=0, label_down=4):
        """

        """
        self.output_size = output_size
        self.pos_ratio = pos_ratio
        self.label_down = label_down

        assert isinstance(self.output_size, (list, tuple))

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        assert (input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i] \
                       for i in range(input_dim)]

        bb_min = [0] * (input_dim + 1)
        bb_max = image.shape
        bb_min, bb_max = bb_min[1:], bb_max[1:]

        ctr = np.array(image.shape[1:]) / 2

        if self.pos_ratio > 0:
            bb_min = ctr - np.array(self.output_size) + 10
            bb_max = ctr + np.array(self.output_size) - 10
            bb_min = np.clip(bb_min, a_min=0, a_max=None).astype('int16')
            bb_max = np.clip(bb_max, a_min=None, a_max=image.shape[1:]).astype('int16')

        crop_min = [random.randint(bb_min[i], max(bb_min[i], bb_max[i] - int(self.output_size[i]))) for i in
                    range(input_dim)]
        crop_min = [min(crop_min[i], input_shape[i + 1] - self.output_size[i]) for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        mask_t = crop_ND_volume_with_bounding_box(mask, crop_min, crop_max)
        sample['image'] = image_t
        sample['mask'] = mask_t
        return sample
