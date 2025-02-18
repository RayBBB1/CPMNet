# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random
from itertools import product
from .utils import compute_bbox3d_intersection_volume

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
        position = sample['position']
        all_loc = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        
        all_rad_pixel = all_rad / image_spacing
        all_nodule_bb_min = all_loc - all_rad_pixel / 2
        all_nodule_bb_max = all_loc + all_rad_pixel / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        
        instance_loc = all_loc[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]

        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        # Generate crop centers
        crop_centers = [*product(z_crop_centers, y_crop_centers, x_crop_centers)]
        crop_centers = np.array(crop_centers)
        
        if self.instance_crop and len(instance_loc) > 0:
            if self.rand_trans is not None:
                instance_crop = instance_loc + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(instance_loc), 3))
            else:
                instance_crop = instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)

        if self.rand_trans is not None:
            crop_centers = crop_centers + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=(len(crop_centers), 3))
        
        all_crop_bb_min = crop_centers - crop_size / 2
        all_crop_bb_min = np.clip(all_crop_bb_min, a_min=0, a_max=shape - crop_size)
        all_crop_bb_min = np.unique(all_crop_bb_min, axis=0)
        
        all_crop_bb_max = all_crop_bb_min + crop_size
        all_crop_bboxes = np.stack([all_crop_bb_min, all_crop_bb_max], axis=1) # [M, 2, 3]
        
        # Compute IoU to determine the label of the patches
        inter_volumes = compute_bbox3d_intersection_volume(all_crop_bboxes, nodule_bboxes) # [M, N]
        all_ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
        max_ious = np.max(all_ious, axis=1) # [M]
        
        tp_indices = max_ious > self.tp_iou
        neg_indices = ~tp_indices

        # Sample patches
        tp_prob = self.tp_ratio / tp_indices.sum() if tp_indices.sum() > 0 else 0
        probs = np.zeros(shape=len(max_ious))
        probs[tp_indices] = tp_prob
        probs[neg_indices] = (1. - probs.sum()) / neg_indices.sum() if neg_indices.sum() > 0 else 0
        probs = probs / probs.sum() # normalize
        sample_indices = np.random.choice(np.arange(len(all_crop_bboxes)), size=self.sample_num, p=probs, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = crop_bb_min + crop_size
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                               crop_bb_min[1]: crop_bb_max[1], 
                               crop_bb_min[2]: crop_bb_max[2]]
            position_crop = position[:, crop_bb_min[0]: crop_bb_max[0],
                                        crop_bb_min[1]: crop_bb_max[1],
                                        crop_bb_min[2]: crop_bb_max[2]]
            image_crop = np.expand_dims(image_crop, axis=0)
            
            ious = all_ious[sample_i] # [N]
            in_idx = np.where(ious > self.tp_iou)[0]
            if in_idx.size > 0:
                # Compute new ctr and rad because of the crop
                all_nodule_bb_min_crop = all_nodule_bb_min - crop_bb_min
                all_nodule_bb_max_crop = all_nodule_bb_max - crop_bb_min
                
                nodule_bb_min_crop = all_nodule_bb_min_crop[in_idx]
                nodule_bb_max_crop = all_nodule_bb_max_crop[in_idx]
                
                nodule_bb_min_crop = np.clip(nodule_bb_min_crop, a_min=0, a_max=None)
                nodule_bb_max_crop = np.clip(nodule_bb_max_crop, a_min=None, a_max=crop_size)
                
                ctr = (nodule_bb_min_crop + nodule_bb_max_crop) / 2
                rad = nodule_bb_max_crop - nodule_bb_min_crop
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])

            sample = dict()
            sample['image'] = np.concatenate([image_crop, position_crop], axis=0) # [C, D, H, W]
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples