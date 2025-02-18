# -*- coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import random
from itertools import product
from .utils import ALL_HARD_FP_RAD, ALL_HARD_FP_LOC, ALL_HARD_FP_PROB, ALL_CLS, ALL_RAD, ALL_LOC, ALL_PROB, ALL_IOU, compute_bbox3d_iou

HARD_TP_THRESHOLD = 0.7

class InstanceCrop(object):
    def __init__(self, crop_size, overlap_ratio: float = 0.25, rand_trans=None, rand_rot=None, instance_crop=True, 
                 tp_ratio=0.7, fp_ratio = 0.2, sample_num=2, blank_side=0, sample_cls=[0]):
        self.sample_cls = sample_cls
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.overlap_ratio = overlap_ratio
        self.overlap_size = (self.crop_size * self.overlap_ratio).astype(np.int32)
        self.stride_size = self.crop_size - self.overlap_size
        
        self.tp_ratio = tp_ratio
        self.fp_ratio = fp_ratio
        self.sample_num = sample_num
        self.blank_side = blank_side
        self.instance_crop = instance_crop

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
    
    def __call__(self, sample, image_spacing: np.ndarray, hard_fp_prob_threshold=None):
        image = sample['image'].astype('float32')
        all_loc = sample[ALL_LOC]
        all_rad = sample[ALL_RAD]
        all_cls = sample[ALL_CLS]
        
        if ALL_HARD_FP_LOC in sample:
            # Hard FP samples
            hard_fp_all_prob = sample[ALL_HARD_FP_PROB]
            valid_mask = hard_fp_all_prob > hard_fp_prob_threshold
            
            hard_fp_all_prob = hard_fp_all_prob[valid_mask]
            hard_fp_all_loc = sample[ALL_HARD_FP_LOC][valid_mask]
            hard_fp_all_rad = sample[ALL_HARD_FP_RAD][valid_mask]
            
            if valid_mask.sum() == 0:
                hard_fp_sample_score = np.zeros((0,))
            else:
                hard_fp_sample_score = hard_fp_all_prob.copy()
                #0.2 + 0.8 * (hard_fp_all_prob - hard_fp_prob_threshold) / (1 - hard_fp_prob_threshold)
        else:
            hard_fp_all_loc = np.zeros((0, 3))
            hard_fp_all_rad = np.zeros((0, 3))
            hard_fp_all_prob = np.zeros((0,))
            hard_fp_sample_score = np.zeros((0,))
        
        if ALL_PROB in sample:
            # Positive samples
            all_prob = sample[ALL_PROB]
            all_iou = sample[ALL_IOU]

            if len(all_prob) == 0:
                tp_sample_score = np.zeros((0,))
            else:
                tp_sample_score = 0.8 * (1 - all_prob) + 0.2 * (1 - all_iou)
            # tp_sample_score = tp_sample_score
        else:
            all_prob = np.zeros((0,))
            all_iou = np.zeros((0,))
            tp_sample_score = np.zeros((0,))
        
        if len(all_loc) != 0:
            if len(all_prob) == 0:
                tp_instance_loc = all_loc.copy()
            else:
                tp_instance_loc = all_loc[all_prob < HARD_TP_THRESHOLD]
        else:
            tp_instance_loc = []
            
        if len(hard_fp_all_loc) != 0:
            if len(hard_fp_all_prob) == 0:
                hard_fp_fp_instance_loc = hard_fp_all_loc.copy().astype(np.int32)
            else:
                hard_fp_fp_instance_loc = hard_fp_all_loc[hard_fp_all_prob >= HARD_TP_THRESHOLD].astype(np.int32)
        else:
            hard_fp_fp_instance_loc = []
            
        image_itk = sitk.GetImageFromArray(image)
        shape = image.shape
        crop_size = np.array(self.crop_size)

        z_crop_centers = self.get_crop_centers(shape, 0)
        y_crop_centers = self.get_crop_centers(shape, 1)
        x_crop_centers = self.get_crop_centers(shape, 2)
        
        crop_centers = [*product(z_crop_centers, y_crop_centers, x_crop_centers)]
        crop_centers = np.array(crop_centers)
        
        if self.instance_crop and len(tp_instance_loc) > 0:
            instance_crop = tp_instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0)
        
        if self.instance_crop and len(hard_fp_fp_instance_loc) > 0:
            instance_crop = hard_fp_fp_instance_loc
            crop_centers = np.append(crop_centers, instance_crop, axis=0) 

        matrixs = []
        crop_scores = []
        tp_nums = []
        for i in range(len(crop_centers)):
            C = crop_centers[i]
            if self.rand_trans is not None:
                C = C + np.random.randint(low=-self.rand_trans, high=self.rand_trans, size=3)

            O = C - np.array(crop_size) / 2
            Z = O + np.array([crop_size[0] - 1, 0, 0])
            Y = O + np.array([0, crop_size[1] - 1, 0])
            X = O + np.array([0, 0, crop_size[2] - 1])
            matrix = np.array([O, X, Y, Z])
            if self.rand_rot is not None:
                matrix = rand_rot_coord(matrix, [-self.rand_rot[0], self.rand_rot[0]],
                                        [-self.rand_rot[1], self.rand_rot[1]],
                                        [-self.rand_rot[2], self.rand_rot[2]], rot_center=C, p=0.8)
            matrixs.append(matrix)
            # According to the matrixs, we can decide if the crop is foreground or background
            bb_min = np.maximum(matrix[0], 0)
            bb_max = bb_min + crop_size
            
            tp_nums.append(np.sum((all_loc > bb_min).all(axis=1) & (all_loc < bb_max).all(axis=1)))
            if len(tp_sample_score) == 0 and len(hard_fp_sample_score) == 0:
                crop_score = 0
            elif len(tp_sample_score) == 0:
                crop_score = np.sum(hard_fp_sample_score[(hard_fp_all_loc > bb_min).all(axis=1) & (hard_fp_all_loc < bb_max).all(axis=1)])
            elif len(hard_fp_sample_score) == 0:
                crop_score = np.sum(tp_sample_score[(all_loc > bb_min).all(axis=1) & (all_loc < bb_max).all(axis=1)])
            else:
                crop_score = np.sum(tp_sample_score[(all_loc > bb_min).all(axis=1) & (all_loc < bb_max).all(axis=1)]) + \
                             np.sum(hard_fp_sample_score[(hard_fp_all_loc > bb_min).all(axis=1) & (hard_fp_all_loc < bb_max).all(axis=1)])
                
            crop_scores.append(crop_score)
        ## Sample patches
        crop_scores = np.array(crop_scores)
        if (len(tp_sample_score) == 0 and len(hard_fp_sample_score) == 0) or crop_scores.sum() == 0:
            # Sample patches
            tp_nums = np.array(tp_nums)
            tp_idx = tp_nums > 0
            neg_idx = tp_nums == 0
            if tp_idx.sum() > 0:
                tp_pos = self.tp_ratio / tp_idx.sum()
            else:
                tp_pos = 0
            p = np.zeros(shape=tp_nums.shape)
            p[tp_idx] = tp_pos
            p[neg_idx] = (1. - p.sum()) / neg_idx.sum() if neg_idx.sum() > 0 else 0
            p = p * 1 / p.sum()
            sample_indices = np.random.choice(np.arange(len(crop_centers)), size=self.sample_num, p=p, replace=False)
        else:
            has_instance_idx = (crop_scores > 0)
            if len(has_instance_idx) < self.sample_num:
                sample_indices = [np.where(has_instance_idx)[0]]
                # Sample fron non-instance crop
                non_instance_idx = np.where(~has_instance_idx)[0]
                sample_indices.extend(np.random.choice(non_instance_idx, size=self.sample_num - len(sample_indices), replace=False))
            else:
                # Normalized
                crop_scores[crop_scores == 0] = np.min(crop_scores[crop_scores != 0]) * 0.1
                crop_scores = crop_scores / crop_scores.sum()
                sample_indices = np.random.choice(np.arange(len(crop_centers)), size=self.sample_num, p=crop_scores, replace=False)
        
        # Crop patches
        samples = []
        for sample_i in sample_indices:
            space = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            matrix = matrixs[sample_i]
            matrix = matrix[:, ::-1]  # in itk axis
            
            image_itk_crop = reorient(image_itk, matrix, spacing=list(space), interp1=sitk.sitkLinear)
            all_loc_crop = [image_itk_crop.TransformPhysicalPointToContinuousIndex(c.tolist()[::-1])[::-1] for c in
                            all_loc]
            all_loc_crop = np.array(all_loc_crop)
            in_idx = []
            for j in range(all_loc_crop.shape[0]):
                if (all_loc_crop[j] <= np.array(image_itk_crop.GetSize()[::-1])).all() and (
                        all_loc_crop[j] >= np.zeros([3])).all():
                    in_idx.append(True)
                else:
                    in_idx.append(False)
            in_idx = np.array(in_idx)

            if in_idx.size > 0:
                ctr = all_loc_crop[in_idx]
                rad = all_rad[in_idx]
                cls = all_cls[in_idx]
            else:
                ctr = np.array([]).reshape(-1, 3)
                rad = np.array([])
                cls = np.array([])

            image_crop = sitk.GetArrayFromImage(image_itk_crop)
            CT_crop = np.expand_dims(image_crop, axis=0)
            shape = np.array(CT_crop.shape[1:])
            if len(rad) > 0:
                rad = rad / image_spacing  # convert pixel coord
            sample = dict()
            sample['image'] = CT_crop
            sample['ctr'] = ctr
            sample['rad'] = rad
            sample['cls'] = cls
            sample['spacing'] = image_spacing
            samples.append(sample)
        return samples

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