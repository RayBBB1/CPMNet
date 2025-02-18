# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict
from .utils import load_series_list, load_image, load_label, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, normalize_processed_image, normalize_raw_image, DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class TrainDataset(Dataset):
    """Dataset for loading numpy images with dimension order [D, H, W]

    Arguments:
        series_list_path (str): Path to the series list file.
        image_spacing (List[float]): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.
        use_bg (bool, optional): Flag indicating whether to use background or not.

    Attributes:
        labels (List): List of labels.
        dicom_paths (List): List of DICOM file paths.
        series_list_path (str): Path to the series list file.
        series_names (List): List of series names.
        image_spacing (ndarray): Spacing of the image in the order [z, y, x].
        transform_post (optional): Transform object to be applied after cropping.
        crop_fn (optional): Cropping function.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, use_bg=False, min_d=0, norm_method='scale'):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        self.series_infos = load_series_list(series_list_path)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d)
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                continue
            
            self.dicom_paths.append(dicom_path)
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn

    def __len__(self):
        return len(self.labels)
    
    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode='c')
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        
        data = {}
        data['image'] = image
        data['all_loc'] = label['all_loc'] # z, y, x
        data['all_rad'] = label['all_rad'] # d, h, w
        data['all_cls'] = label['all_cls']
        data['file_name'] = series_name
        samples = self.crop_fn(data, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'], DEFAULT_WINDOW_LEVEL, DEFAULT_WINDOW_WIDTH)
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            sample['spacing'] = image_spacing
            if self.transform_post:
                sample['ctr_transform'] = []
                sample['feat_transform'] = []
                sample = self.transform_post(sample)
            random_samples.append(sample)

        return random_samples

class DetDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, image_spacing: List[float], SplitComb, norm_method='scale'):
        self.series_list_path = series_list_path
        
        self.labels = []
        self.dicom_paths = []
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.series_infos = load_series_list(series_list_path)
        
        for folder, series_name in self.series_infos:
            dicom_path = gen_dicom_path(folder, series_name)
            self.dicom_paths.append(dicom_path)
        self.splitcomb = SplitComb
        
            
            
            
    def __len__(self):
        return len(self.dicom_paths)
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        
        image_spacing = self.image_spacing.copy() # z, y, x
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)

        data = {}
        # split_images [N, 1, crop_z, crop_y, crop_x]
        split_images, nzhw = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data

class ReFineSplitComb():
    def __init__(self, crop_size: List[int]=[96, 96, 96]):
        self.crop_size = np.array(crop_size, dtype=np.int32)
        self.topk = 1
        
    def split(self, image, cand_infos):
        crop_images = []
        nodule_centers = []
        nodule_shapes = []
        crop_bb_mins = []
        
        image_shape = np.array(image.shape, dtype=np.int32)
        for cand_info in cand_infos:
            coordX, coordY, coordZ, prob, w, h, d = cand_info
            nodule_center = np.array([coordZ, coordY, coordX], dtype=np.float32)
            nodule_shape = np.array([d, h, w], dtype=np.float32)
            
            crop_bb_min = np.round(nodule_center - self.crop_size / 2).astype(np.int32)
            crop_bb_min = np.clip(crop_bb_min, 0, image_shape - self.crop_size)
            crop_bb_max = crop_bb_min + self.crop_size
            
            crop_nodule_center = nodule_center - crop_bb_min
            crop_image = image[crop_bb_min[0]:crop_bb_max[0], 
                                crop_bb_min[1]:crop_bb_max[1], 
                                crop_bb_min[2]:crop_bb_max[2]]
                
            crop_images.append(crop_image)
            crop_bb_mins.append(crop_bb_min)
            nodule_centers.append(crop_nodule_center)
            nodule_shapes.append(nodule_shape)
            
        return crop_images, crop_bb_mins, nodule_centers, nodule_shapes

    def combine(self, outputs, crop_bb_mins, nodule_centers, nodule_shapes):
        """
        Args:
            outputs: [N, M, 8], 8-> id, prob, z_min, y_min, x_min, d, h, w
            crop_bb_mins: [N, 3], 3-> z_min, y_min, x_min
            nodule_centers: [N, 3], 3-> z, y, x
            nodule_shapes: [N, 3], 3-> d, h, w
        """
        outputs[..., 2:5] += np.expand_dims(crop_bb_mins, axis=1)
        nodule_centers = nodule_centers + crop_bb_mins
        
        dist = np.linalg.norm(outputs[..., 2:5] - np.expand_dims(nodule_centers, axis=1), axis=-1) # [N, M]
        # Keep top k
        keep_outputs = []
        for i in range(dist.shape[0]):
            topk_idx = np.argsort(dist[i])[:self.topk]
            keep_outputs.extend(outputs[i, topk_idx])
        keep_outputs = np.concatenate(keep_outputs, axis=0)
        return keep_outputs
    
class DetRefinedDataset(Dataset):
    """Detection dataset for inference
    """
    def __init__(self, series_list_path: str, series_cands: Dict[str, np.ndarray], SplitComb, norm_method='scale'):
        self.series_list_path = series_list_path
        
        self.norm_method = norm_method
        self.series_cands = series_cands
        self.series_names = list(series_cands.keys())
        self.splitcomb = SplitComb
        
        self.series_infos = load_series_list(series_list_path)
        self.dicom_paths = dict()
        for folder, series_name in self.series_infos:
            dicom_path = gen_dicom_path(folder, series_name)
            self.dicom_paths[series_name] = dicom_path
            
    @staticmethod
    def load_series_cands(series_cands_path: str) -> Dict[str, np.ndarray]:
        with open(series_cands_path, 'r') as file:
            lines = file.readlines()[1:]
        prediction = defaultdict(list)
        for line in lines:
            line = line.split(',')
            series_id = line[0]
            line = [float(x.strip()) for x in line[1:]]
            coordX, coordY, coordZ, prob, w, h, d = line
            prediction[series_id].append([coordX, coordY, coordZ, prob, w, h, d])
        return prediction
    
    def __getitem__(self, idx):
        series_name = self.series_names[idx]
        cands = self.series_cands[series_name]
        
        dicom_path = self.dicom_paths[series_name]
        image = load_image(dicom_path) # z, y, x
        image = normalize_processed_image(image, self.norm_method)
        
        data = {'series_name': series_name, 
                'series_path': dicom_path}
        crop_images, crop_bb_mins, nodule_centers, nodule_shapes = self.splitcomb.split(image, cands)
        
        data['crop_images'] = np.expand_dims(np.ascontiguousarray(crop_images), axis=1)
        data['crop_bb_mins'] = np.array(crop_bb_mins)
        data['nodule_centers'] = np.array(nodule_centers)
        data['nodule_shapes'] = np.array(nodule_shapes)
        return data
    
    def __len__(self):
        return len(self.series_cands)