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
        rand_trans (list[int], optional): The range of random translation. Defaults to None.
        rand_rot (list[int], optional): The range of random rotation. Defaults to None.
        instance_crop (bool, optional): Whether to perform additional sampling with instance around the center. Defaults to True.
        overlap_size (list[int], optional): The size of overlap of sliding window. Defaults to [16, 32, 32].
        tp_ratio (float, optional): The sampling rate for a patch containing at least one lesion. Defaults to 0.7.
        sample_num (int, optional): The number of patches per CT. Defaults to 2.
        blank_side (int, optional): The number of pixels near the patch border where labels are set to ignored. Defaults to 0.
        sample_cls (list[int], optional): The list of classes to sample patches from. Defaults to [0].
    """

    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, instance_crop=True, 
                 tp_ratio=0.7, sample_num=2, blank_side=0, sample_cls=[0], tp_iou = 0.5):
        """This is crop function with spatial augmentation for training Lesion Detection.

        Arguments:
            crop_size: patch size
            rand_trans: random translation
            rand_rot: random rotation
            instance_crop: additional sampling with instance around center
            spacing: output patch spacing, [z,y,x]
            base_spacing: spacing of the numpy image.
            overlap_size: the size of overlap  of sliding window
            tp_ratio: sampling rate for a patch containing at least one leision
            sample_num: patch number per CT
            blank_side:  labels within blank_side pixels near patch border is set to ignored.
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

        if rand_trans == None:
            self.rand_trans = None
        else:
            self.rand_trans = np.array(rand_trans)

        self.tp_iou = tp_iou
        
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
        all_loc = sample['series_labels']['all_loc']
        all_rad = sample['series_labels']['all_rad']
        all_cls = sample['series_labels']['all_cls']
        
        all_nodule_bb_min = all_loc - all_rad / 2
        all_nodule_bb_max = all_loc + all_rad / 2
        nodule_bboxes = np.stack([all_nodule_bb_min, all_nodule_bb_max], axis=1) # [N, 2, 3]
        nodule_volumes = np.prod(all_rad, axis=1) # [N]
        
        target_loc = sample['target']['all_loc'][0]
        
        shape = image.shape
        crop_size = self.crop_size
        
        crop_bb_min = target_loc - crop_size / 2
        crop_bb_min = np.clip(crop_bb_min, a_min=0, a_max=shape - crop_size)
        crop_bb_min = np.unique(crop_bb_min, axis=0)
        
        crop_bb_max = crop_bb_min + crop_size
        crop_bboxes = np.stack([crop_bb_min, crop_bb_max], axis=0)
        crop_bboxes = crop_bboxes[np.newaxis, :]
        # Compute IoU to determine the label of the patches
        inter_volumes = compute_bbox3d_intersection_volume(crop_bboxes, nodule_bboxes) # [M, N]
        ious = inter_volumes / nodule_volumes[np.newaxis, :] # [M, N]
        
        crop_bb_min = crop_bb_min.astype(np.int32)
        crop_bb_max = crop_bb_max.astype(np.int32)
        image_crop = image[crop_bb_min[0]: crop_bb_max[0], 
                            crop_bb_min[1]: crop_bb_max[1], 
                            crop_bb_min[2]: crop_bb_max[2]]
        image_crop = np.expand_dims(image_crop, axis=0)
        
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
            
        if len(rad) > 0:
            rad = rad / image_spacing  # convert pixel coord
        sample = dict()
        sample['image'] = image_crop
        sample['ctr'] = ctr
        sample['rad'] = rad
        sample['cls'] = cls
        sample['spacing'] = image_spacing
        return sample

def rotate_vecs_3d(vec, angle, axis):
    rad = np.deg2rad(angle)
    rotated_vec = vec.copy()
    rotated_vec[::, axis[0]] = vec[::, axis[0]] * np.cos(rad) - vec[::, axis[1]] * np.sin(rad)
    rotated_vec[::, axis[1]] = vec[::, axis[0]] * np.sin(rad) + vec[::, axis[1]] * np.cos(rad)
    return rotated_vec

def apply_transformation_coord(coord, transform_param_list, rot_center):
    """
    apply rotation transformation to an ND image
    Args:
        image (nd array): the input nd image
        transform_param_list (list): a list of roration angle and axes
        order (int): interpolation order
    """
    for angle, axes in transform_param_list:
        # rot_center = np.random.uniform(low=np.min(coord, axis=0), high=np.max(coord, axis=0), size=3)
        org = coord - rot_center
        new = rotate_vecs_3d(org, angle, axes)
        coord = new + rot_center

    return coord

def rand_rot_coord(coord, angle_range_d, angle_range_h, angle_range_w, rot_center, p):
    transform_param_list = []

    if (angle_range_d[1]-angle_range_d[0] > 0) and (random.random() < p):
        angle_d = np.random.uniform(angle_range_d[0], angle_range_d[1])
        transform_param_list.append([angle_d, (-2, -1)])
    if (angle_range_h[1]-angle_range_h[0] > 0) and (random.random() < p):
        angle_h = np.random.uniform(angle_range_h[0], angle_range_h[1])
        transform_param_list.append([angle_h, (-3, -1)])
    if (angle_range_w[1]-angle_range_w[0] > 0) and (random.random() < p):
        angle_w = np.random.uniform(angle_range_w[0], angle_range_w[1])
        transform_param_list.append([angle_w, (-3, -2)])

    if len(transform_param_list) > 0:
        coord = apply_transformation_coord(coord, transform_param_list, rot_center)

    return coord

def reorient(itk_img, mark_matrix, spacing=[1., 1., 1.], interp1=sitk.sitkLinear):
    '''
    itk_img: image to reorient
    mark_matric: physical mark point
    '''
    spacing = spacing[::-1]
    origin, x_mark, y_mark, z_mark = np.array(mark_matrix[0]), np.array(mark_matrix[1]), np.array(
        mark_matrix[2]), np.array(mark_matrix[3])

    # centroid_world = itk_img.TransformContinuousIndexToPhysicalPoint(centroid)
    filter_resample = sitk.ResampleImageFilter()
    filter_resample.SetInterpolator(interp1)
    filter_resample.SetOutputSpacing(spacing)

    # set origin
    origin_reorient = mark_matrix[0]
    # set direction
    # !!! note: column wise
    x_base = (x_mark - origin) / np.linalg.norm(x_mark - origin)
    y_base = (y_mark - origin) / np.linalg.norm(y_mark - origin)
    z_base = (z_mark - origin) / np.linalg.norm(z_mark - origin)
    direction_reorient = np.stack([x_base, y_base, z_base]).transpose().reshape(-1).tolist()

    # set size
    x, y, z = np.linalg.norm(x_mark - origin) / spacing[0], np.linalg.norm(y_mark - origin) / spacing[
        1], np.linalg.norm(z_mark - origin) / spacing[2]
    size_reorient = (int(np.ceil(x + 0.5)), int(np.ceil(y + 0.5)), int(np.ceil(z + 0.5)))

    filter_resample.SetOutputOrigin(origin_reorient)
    filter_resample.SetOutputDirection(direction_reorient)
    filter_resample.SetSize(size_reorient)
    # filter_resample.SetSpacing([sp]*3)

    filter_resample.SetOutputPixelType(itk_img.GetPixelID())
    itk_out = filter_resample.Execute(itk_img)

    return itk_out