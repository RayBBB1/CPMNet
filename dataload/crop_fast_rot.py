# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random
import math
import torch
from scipy import ndimage as nd
from itertools import product
from .utils import get_HU_MIN_MAX, DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH

def compute_bbox3d_intersection_volume(box1: np.ndarray, box2: np.ndarray):
    """ 
    Args:
        box1 (shape = [N, 2, 3])
        box2 (shape = [M, 2, 3])
    Return:
        the area of the intersection between box1 and box2, shape = [N, M]
    """
    a1, a2 = box1[:,np.newaxis, 0,:], box1[:,np.newaxis, 1,:] # [N, 1, 3]
    b1, b2 = box2[np.newaxis,:, 0,:], box2[np.newaxis,:, 1,:] # [1, N, 3]
    inter_volume = np.clip((np.minimum(a2, b2) - np.maximum(a1, b1)),0, None).prod(axis=2)

    return inter_volume

class InstanceCrop(object):
    """Randomly crop the input image (shape [C, D, H, W]
    """

    def __init__(self, crop_size, rand_trans=None, rand_rot=None, instance_crop=True, overlap_ratio: float = 0.25, 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou = 0.7):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            instance_crop: additional sampling with instance around center
            overlap_size: the size of overlap  of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.
            sample_cls: the class of the sample
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
        
        self.tp_iou = 0.0 #TODO

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)
            
        if rand_rot == None:
            self.rand_rot = None
        else:
            self.rand_rot = np.array(rand_rot)
            self.rot_crop_size = ((np.sqrt(2) * self.crop_size).astype(np.int32) - 1) // 2 * 2 # round to even

        self.use_gpu = torch.cuda.is_available()
        self.hu_min = get_HU_MIN_MAX(DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH)[0]
        
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
    
    @staticmethod
    def random_rotate_points(bbox_points: np.ndarray, image_shape: np.ndarray, angle: int):
        """
        Args:
            bbox_points (np.ndarray): 4 different points of the bounding box with shape (4, 2) in order of (left-top, right-top, right-bottom, left-bottom).
            image_shape (np.ndarray): 2D shape of the image.
            angle (int): The maximum angle (in degrees) by which to rotate the points.

        Returns:
            np.ndarray: The rotated 2D points.
        """
        rot_angles = np.random.randint(-angle, angle, size=(len(bbox_points), 1))
        radian = np.deg2rad(rot_angles)
        cos = np.cos(radian)
        sin = np.sin(radian)
        
        for i in range(len(bbox_points.shape) - len(cos.shape)):
            cos = np.expand_dims(cos, axis=1)
            sin = np.expand_dims(sin, axis=1)
        
        img_center = np.array(image_shape) / 2
        rotated_bbox_points = bbox_points.copy()
        
        rotated_bbox_points[..., 0:1] = (bbox_points[..., 0:1] - img_center[0]) * cos - (bbox_points[..., 1:2] - img_center[1]) * sin + img_center[0]
        rotated_bbox_points[..., 1:2] = (bbox_points[..., 0:1] - img_center[0]) * sin + (bbox_points[..., 1:2] - img_center[1]) * cos + img_center[1]
        return rotated_bbox_points, rot_angles
    
    def __call__(self, sample, image_spacing: np.ndarray):
        image = sample['image']
        all_nodule_ctrs = sample['all_loc']
        all_rad = sample['all_rad']
        all_cls = sample['all_cls']
        
        all_rad_pixel = all_rad / image_spacing
        all_nodule_bb_min = all_nodule_ctrs - all_rad_pixel / 2
        all_nodule_bb_max = all_nodule_ctrs + all_rad_pixel / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad_pixel, axis=1) # [N]
        
        instance_loc = all_nodule_ctrs[np.sum([all_cls == cls for cls in self.sample_cls], axis=0, dtype='bool')]

        image_shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(image_shape, 0)
        y_crop_centers = self.get_crop_centers(image_shape, 1)
        x_crop_centers = self.get_crop_centers(image_shape, 2)
        
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
        
        if self.rand_rot is not None:
            rot_crop_ctrs = np.clip(crop_centers, a_min=self.rot_crop_size / 2, a_max=image_shape - self.rot_crop_size / 2)
            rot_crop_ctrs = np.unique(rot_crop_ctrs, axis=0)
            
            all_rot_bb_min = (rot_crop_ctrs - self.rot_crop_size / 2).astype(np.int32) # [M, 3]
            
            all_rot_nodule_bb_min = np.expand_dims(all_nodule_bb_min, axis=0) - np.expand_dims(all_rot_bb_min, axis=1) # [M, N, 3]
            all_rot_nodule_bb_max = np.expand_dims(all_nodule_bb_max, axis=0) - np.expand_dims(all_rot_bb_min, axis=1)
            
            #TODO
            # Now only rotate y-x plane
            all_left_top = all_rot_nodule_bb_min.copy()
            all_right_top = np.stack([all_rot_nodule_bb_min[:, :, 0], all_rot_nodule_bb_min[:, :, 1], all_rot_nodule_bb_max[:, :, 2]], axis=2)
            all_right_bottom = np.stack([all_rot_nodule_bb_min[:, :, 0], all_rot_nodule_bb_max[:, :, 1], all_rot_nodule_bb_max[:, :, 2]], axis=2)
            all_left_bottom = np.stack([all_rot_nodule_bb_min[:, :, 0], all_rot_nodule_bb_max[:, :, 1], all_rot_nodule_bb_min[:, :, 2]], axis=2)

            all_bbox_points = np.stack([all_left_top, all_right_top, all_right_bottom, all_left_bottom], axis=2) # [M, N, 4, 3]
            # yx_image_shape = image_shape[1:3]
            all_rot_bbox_points, all_rot_angles = self.random_rotate_points(all_bbox_points[...,1:].copy(), self.rot_crop_size[1:].copy(), self.rand_rot[0])
            
            # Convert to new bbox
            # all_rot_bbox_points is in shape [M, N, 4, 2]
            all_rot_nodule_bb_min[:, :, 1] = np.min(all_rot_bbox_points[..., 0], axis=2) # [M, N, 3]
            all_rot_nodule_bb_min[:, :, 2] = np.min(all_rot_bbox_points[..., 1], axis=2) # [M, N, 3]
    
            all_rot_nodule_bb_max[:, :, 1] = np.max(all_rot_bbox_points[..., 0], axis=2) # [M, N, 3]
            all_rot_nodule_bb_max[:, :, 2] = np.max(all_rot_bbox_points[..., 1], axis=2) # [M, N, 3]
            
            # all_rot_angles # [M, N, 1]
            # refine_offset = (all_rot_nodule_bb_max - all_rot_nodule_bb_min) * np.expand_dims(((np.abs(all_rot_angles) / 45) * 0.1), axis=2)
            # refine_offset = refine_offset[:, :, 1:] # [M, N, 2]
            
            # larger_axis = np.argmax(refine_offset, axis=2) # [M, N]
            # larger_indices = np.indices(larger_axis.shape)
            # larger_indices = np.stack([larger_indices[0], larger_indices[1], larger_axis], axis=2)
            
            # aspect_ratio = refine_offset[..., 0] / np.maximum(refine_offset[..., 1], 1e-6) # [M, N]
            # aspect_ratio = np.where(aspect_ratio < 1, 1 / np.maximum(aspect_ratio, 1e-6), aspect_ratio)
            
            # new_refine_offset = np.zeros_like(refine_offset)
            # new_refine_offset[larger_indices[..., 0], larger_indices[..., 1], larger_indices[..., 2]] = refine_offset[larger_indices[..., 0], larger_indices[..., 1], larger_indices[..., 2]]
            # new_refine_offset *= np.expand_dims(np.maximum(np.minimum(aspect_ratio, 2), 1), axis=2)
            
            # all_rot_nodule_bb_min[..., 1:] = all_rot_nodule_bb_min[..., 1:] + new_refine_offset
            # all_rot_nodule_bb_max[..., 1:] = all_rot_nodule_bb_max[..., 1:] - new_refine_offset
            
            all_rot_nodule_bb_rad = all_rot_nodule_bb_max - all_rot_nodule_bb_min
            all_rot_nodule_bb_volume = np.prod(all_rot_nodule_bb_rad, axis=2) # [M, N]
            
            crop_valid_min = (self.rot_crop_size - self.crop_size) // 2
            valid_range = np.array([crop_valid_min, crop_valid_min + self.crop_size], dtype=np.int32)
            
            all_rot_nodule_bboxes = np.stack([all_rot_nodule_bb_min.reshape(-1, 3), all_rot_nodule_bb_max.reshape(-1, 3)], axis=1)
            inter_volumes = compute_bbox3d_intersection_volume(all_rot_nodule_bboxes, valid_range[np.newaxis, :])
            
            all_ious = inter_volumes / all_rot_nodule_bb_volume.reshape(-1, 1)
            all_ious = np.array(np.split(all_ious, len(rot_crop_ctrs), axis=0)) # [M, N]
            all_ious = np.squeeze(all_ious, axis=2)
            
            all_crop_bb_min = all_rot_bb_min
            all_crop_bb_max = all_rot_bb_min + self.rot_crop_size
        else:
            crop_centers = np.unique(crop_centers, axis=0)
            
            all_crop_bb_min = crop_centers - crop_size / 2
            all_crop_bb_min = np.clip(all_crop_bb_min, a_min=0, a_max=image_shape - crop_size)
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
        sample_indices = np.random.choice(np.arange(len(all_crop_bb_min)), size=self.sample_num, p=probs, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            crop_bb_min = all_crop_bb_min[sample_i].astype(np.int32)
            crop_bb_max = all_crop_bb_max[sample_i].astype(np.int32)
            image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                                crop_bb_min[1]: crop_bb_max[1], 
                                crop_bb_min[2]: crop_bb_max[2]]
            if self.rand_rot is not None:
                angle = all_rot_angles[sample_i][0]
                if self.use_gpu:
                    import cupy as cp
                    from cupyx.scipy import ndimage as cupy_nd
                    image_crop = cp.array(image_crop)
                    image_crop = cupy_nd.rotate(image_crop, angle, axes=(1, 2), reshape=False, order=3, mode='constant', cval=self.hu_min)
                    image_crop = image_crop[valid_range[0, 0]: valid_range[1, 0],
                                            valid_range[0, 1]: valid_range[1, 1],
                                            valid_range[0, 2]: valid_range[1, 2]]
                    image_crop = cp.asnumpy(image_crop)
                else:
                    image_crop = nd.rotate(image_crop, angle, axes=(1, 2), reshape=False, order=3, mode='constant', cval=self.hu_min)
                    image_crop = image_crop[valid_range[0, 0]: valid_range[1, 0],
                                        valid_range[0, 1]: valid_range[1, 1],
                                        valid_range[0, 2]: valid_range[1, 2]]
            image_crop = np.expand_dims(image_crop, axis=0)
            
            ious = all_ious[sample_i] # [N]
            in_idx = np.where(ious > self.tp_iou)[0]
            if in_idx.size > 0:
                if self.rand_rot is not None:
                    rot_nodule_bb_min = all_rot_nodule_bb_min[sample_i][in_idx]
                    rot_nodule_bb_max = all_rot_nodule_bb_max[sample_i][in_idx]
                    
                    rot_nodule_bb_min = rot_nodule_bb_min - valid_range[0]
                    rot_nodule_bb_max = rot_nodule_bb_max - valid_range[0]
                    
                    # rot_nodule_bb_min = np.clip(rot_nodule_bb_min, a_min=0, a_max=None)
                    # rot_nodule_bb_max = np.clip(rot_nodule_bb_max, a_min=None, a_max=self.crop_size)
                    
                    ctr = (rot_nodule_bb_min + rot_nodule_bb_max) / 2
                    rad = rot_nodule_bb_max - rot_nodule_bb_min
                    cls = all_cls[in_idx]
                else:
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
            sample['image'] = image_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples