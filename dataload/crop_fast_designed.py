# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import random
from itertools import product
from .utils import compute_bbox3d_intersection_volume

def get_random_offset(upper: np.ndarray, num: int, lower = [0, 0, 0]):
    offset =  np.random.randint(low=lower, high=upper, size=(num, 3))
    # Times -1 with 50% probability
    offset = offset * np.random.choice([-1, 1], size=(num, 3))
    return offset

class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W])

    Args:
        crop_size (list[int]): The size of the patch to be cropped.
        rand_trans (list[int], optional): Random translation. Defaults to None.
        instance_crop (bool, optional): Flag to enable additional sampling with instance around center. Defaults to True.
        overlap_size (list[int], optional): The size of overlap of sliding window. Defaults to [16, 32, 32].
        tp_ratio (float, optional): Sampling rate for a patch containing at least one lesion. Defaults to 0.7.
        sample_num (int, optional): Patch number per CT. Defaults to 2.
        blank_side (int, optional): Labels within blank_side pixels near patch border is set to ignored. Defaults to 0.
        sample_cls (list[int], optional): The class of the sample. Defaults to [0].
        tp_iou (float, optional): IoU threshold to determine the label of the patches. Defaults to 0.5.
    """
    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, instance_crop=True, 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou=0.5):
        """This is crop function with spatial augmentation for training Lesion Detection.
        """
        self.sample_cls = sample_cls
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.overlap_ratio = overlap_ratio
        self.overlap_size = (self.crop_size * self.overlap_ratio).astype(np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.tp_ratio = tp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop
        self.tp_iou = tp_iou

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        if rand_rot == None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)

    def get_crop_centers(self, shape, dim: int):
        crop = self.crop_size[dim]
        overlap = self.overlap_size[dim]
        stride = self.stride_size[dim]
        shape = shape[dim]
        
        crop_centers = np.arange(0, shape - overlap, stride) + crop / 2
        crop_centers = np.clip(crop_centers, a_max=shape - crop / 2, a_min=None)
        
        # Add final center
        crop_centers = np.append(crop_centers, shape - crop / 2)
        crop_centers = np.unique(crop_centers)
        
        return crop_centers
    
    def __call__(self, sample, image_spacing: np.ndarray):
        image = sample['image']
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        
        if len(all_rad) != 0:
            all_rad_pixel = all_rad / image_spacing
            all_nodule_bb_min = all_loc - all_rad_pixel / 2
            all_nodule_bb_max = all_loc + all_rad_pixel / 2
            nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
            nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        else:
            nodule_bboxes = np.zeros((0, 2, 3))
            nodule_volumes = np.zeros(0)
        
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]

        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        # Generate crop centers
        crop_centers = np.array([*product(z_crop_centers, y_crop_centers, x_crop_centers)])
        bg_centers = []
        bg_centers.append(crop_centers + get_random_offset(self.rand_trans, len(crop_centers)))
        bg_centers.append(crop_centers + get_random_offset((self.rand_trans * 1.5).astype(np.int32), len(crop_centers), lower=(self.rand_trans * 0.5).astype(np.int32)))
        
        bg_centers = np.concatenate(bg_centers, axis=0)
        bg_bboxes_min = bg_centers - crop_size / 2
        bg_bboxes_min = np.clip(bg_bboxes_min, a_min=0, a_max=shape - crop_size)
        bg_bboxes_min = np.unique(bg_bboxes_min, axis=0)
        bg_bboxes_max = bg_bboxes_min + crop_size
        
        bg_bboxes = np.stack([bg_bboxes_min, bg_bboxes_max], axis=1) # [M, 2, 3]
        
        # Compute IoU to determine the label of the patches
        if len(nodule_bboxes) != 0:
            inter_volumes = compute_bbox3d_intersection_volume(bg_bboxes, nodule_bboxes) # [M, N]
            all_ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
            max_ious = np.max(all_ious, axis=1) # [M]
        else:
            all_ious = np.zeros((len(bg_bboxes), 1))
            max_ious = np.zeros(len(bg_bboxes))
        
        bg_bboxes_min = bg_bboxes_min[max_ious <= 0.1]
        
        if len(nodule_bboxes) == 0:
            num_fg = 0
            num_bg = self.sample_num 
        elif len(bg_bboxes_min) == 0:
            num_fg = self.sample_num
            num_bg = 0
        else:
            num_fg = int(self.sample_num * self.tp_ratio)
            num_bg = self.sample_num - num_fg
            
        all_crop_bb_min = []
        # Sample fg patches
        if num_fg > 0:
            fg_centers = instance_loc.copy()
            if len(fg_centers) > num_fg:
                fg_bboxes_min = fg_centers + get_random_offset(self.crop_size // 2, len(fg_centers)) - crop_size // 2
                fg_bboxes_min = np.clip(fg_bboxes_min, a_min=0, a_max=shape - crop_size)
                
                fg_indices = np.random.choice(np.arange(len(fg_centers)), size=num_fg, replace=False)
                all_crop_bb_min.append(fg_bboxes_min[fg_indices])
            else:
                # Simple case
                fg_bboxes_min = fg_centers + get_random_offset(self.crop_size // 4, len(fg_centers)) - crop_size // 2
                fg_bboxes_min = np.clip(fg_bboxes_min, a_min=0, a_max=shape - crop_size)
                all_crop_bb_min.append(fg_bboxes_min)
                # Hard case
                num_remain = num_fg - len(fg_centers)
                cur_i = 0
                random_order = np.random.permutation(len(fg_centers))
                while num_remain > 0:
                    center = fg_centers[random_order[cur_i]]
                    center = center + get_random_offset(self.crop_size // 2 - np.random.randint(2, 10), 1, lower=self.crop_size // 3)
                    bbox_min = center - crop_size // 2
                    bbox_min = np.clip(bbox_min, a_min=0 + np.random.randint(2, 10), a_max=shape - crop_size - np.random.randint(2, 10))
                    all_crop_bb_min.append(bbox_min)
                    cur_i = (cur_i + 1) % len(fg_centers)
                    num_remain = num_remain - 1
        
        # Sample bg patches
        if num_bg > 0:
            if num_bg > len(bg_bboxes_min):
                replace = True
            else:
                replace = False
            bg_indices = np.random.choice(np.arange(len(bg_bboxes_min)), size=num_bg, replace=replace)
            all_crop_bb_min.append(bg_bboxes_min[bg_indices])
    
        all_crop_bb_min = np.concatenate(all_crop_bb_min, axis=0)
        all_crop_bb_max = all_crop_bb_min + crop_size
        all_crop_bboxes = np.stack([all_crop_bb_min, all_crop_bb_max], axis=1)
        # Compute IoU to determine the label of the patches
        if len(nodule_bboxes) != 0:
            inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, nodule_bboxes)
            all_ious = inter_volumes / nodule_volumes[np.newaxis, :]
            max_ious = np.max(all_ious, axis=1)
        else:
            all_ious = np.zeros((len(all_crop_bboxes), 1))
            max_ious = np.zeros(len(all_crop_bboxes))
            
        # Crop patches
        samples = []
        for i in range(len(all_crop_bb_min)):
            crop_bb_min = all_crop_bb_min[i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                               crop_bb_min[1]: crop_bb_max[1], 
                               crop_bb_min[2]: crop_bb_max[2]]
            image_crop = np.expand_dims(image_crop, axis=0)
            
            ious = all_ious[i] # [N]
            in_idx = np.where(ious > 0)[0]
            if in_idx.size > 0:
                # Compute new ctr and rad because of the crop
                all_nodule_bb_min_crop = all_nodule_bb_min - crop_bb_min
                all_nodule_bb_max_crop = all_nodule_bb_max - crop_bb_min
                
                nodule_bb_min_crop = all_nodule_bb_min_crop[in_idx]
                nodule_bb_max_crop = all_nodule_bb_max_crop[in_idx]
                
                ctr = (nodule_bb_min_crop + nodule_bb_max_crop) / 2
                rad = nodule_bb_max_crop - nodule_bb_min_crop
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])

            sample = dict()
            sample['image'] = image_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples