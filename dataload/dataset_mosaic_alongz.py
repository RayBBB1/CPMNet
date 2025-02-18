# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import numpy as np
from typing import List
from .utils import load_series_list, load_image, load_label, ALL_RAD, ALL_LOC, ALL_CLS, gen_dicom_path, gen_label_path, normalize_processed_image, normalize_raw_image
from torch.utils.data import Dataset
from itertools import product

logger = logging.getLogger(__name__)

def coordToAnnot(sample):
    ctr = sample['ctr']
    rad = sample['rad']
    cls = sample['cls']
    
    spacing = sample['spacing']
    n = ctr.shape[0]
    spacing = np.tile(spacing, (n, 1))
    
    annot = np.concatenate([ctr, rad.reshape(-1, 3), spacing.reshape(-1, 3), cls.reshape(-1, 1)], axis=-1).astype('float32') # (n, 10)

    sample['annot'] = annot
    return sample

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
    def __init__(self, series_list_path: str, image_spacing: List[float], transform_post=None, crop_fn=None, 
                 use_bg=False, min_d=0, min_size: int = 0, norm_method='scale', mmap_mode=None):
        self.labels = []
        self.dicom_paths = []
        self.series_list_path = series_list_path
        self.norm_method = norm_method
        self.image_spacing = np.array(image_spacing, dtype=np.float32) # (z, y, x)
        self.min_d = int(min_d)
        self.min_size = int(min_size)
        
        if self.min_d > 0:
            logger.info('When training, ignore nodules with depth less than {}'.format(min_d))
        if self.min_size != 0:
            logger.info('When training, ignore nodules with size less than {}'.format(min_size))
        
        if self.norm_method == 'mean_std':
            logger.info('Normalize image to have mean 0 and std 1, and then scale to -1 to 1')
        elif self.norm_method == 'scale':
            logger.info('Normalize image to have value ranged from -1 to 1')
        elif self.norm_method == 'none':
            logger.info('Normalize image to have value ranged from 0 to 1')
        
        if use_bg:
            logger.info('Using background(healthy lung) as training data')
        
        self.series_infos = load_series_list(series_list_path)
        for folder, series_name in self.series_infos:
            label_path = gen_label_path(folder, series_name)
            dicom_path = gen_dicom_path(folder, series_name)
           
            label = load_label(label_path, self.image_spacing, min_d, min_size)
            if label[ALL_LOC].shape[0] == 0 and not use_bg:
                continue
            
            self.dicom_paths.append(dicom_path)
            self.labels.append(label)

        self.transform_post = transform_post
        self.crop_fn = crop_fn
        self.mmap_mode = mmap_mode

    def __len__(self):
        return len(self.labels)
    
    def load_image(self, dicom_path: str) -> np.ndarray:
        """
        Return:
            A 3D numpy array with dimension order [D, H, W] (z, y, x)
        """
        image = np.load(dicom_path, mmap_mode=self.mmap_mode)
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def __getitem__(self, idx):
        dicom_path = self.dicom_paths[idx]
        series_folder = self.series_infos[idx][0]
        series_name = self.series_infos[idx][1]
        label = self.labels[idx]

        image_spacing = self.image_spacing.copy() # z, y, x
        image = self.load_image(dicom_path) # z, y, x
        
        samples = {}
        samples['image'] = image
        samples[ALL_LOC] = label[ALL_LOC] # z, y, x
        samples[ALL_RAD] = label[ALL_RAD] # d, h, w
        samples[ALL_CLS] = label[ALL_CLS]
        samples['file_name'] = series_name
        samples = self.crop_fn(samples, image_spacing)
        random_samples = []

        for i in range(len(samples)):
            sample = samples[i]
            sample['image'] = normalize_raw_image(sample['image'])
            sample['image'] = normalize_processed_image(sample['image'], self.norm_method)
            if self.transform_post:
                sample['ctr_transform'] = []
                sample['feat_transform'] = []
                sample = self.transform_post(sample)
            random_samples.append(sample)

        # Mosaic
        crop_size = np.array(random_samples[0]['image'].shape[1:], dtype=np.int32)
        
        start_points = np.array([[0,0,0],[crop_size[0], 0, 0]])
        end_points = start_points + crop_size
        
        mosaic_samples = []
        for sample_i in range(3):
            image = np.zeros([crop_size[0] * 2, crop_size[1], crop_size[2]], dtype=np.float32)
            image = np.expand_dims(image, axis=0) # Add channel dimension, shape: (1, 2*crop_z, 2*crop_y, 2*crop_x)
            mosaic_sample = {'ctr_transform': [], 
                            'feat_transform': [], 
                            'ctr': [],
                            'rad': [],
                            'cls': [],
                            'spacing': np.array([1.0, 1.0, 1.0], dtype=np.float32)}
            for i, (start_point, sample) in enumerate(zip(start_points, random_samples)):
                image[0, start_point[0]:end_points[i][0], start_point[1]:end_points[i][1], start_point[2]:end_points[i][2]] = sample['image']
                if sample['ctr'].shape[0] > 0:
                    sample['ctr'] = sample['ctr'] + start_point
                    mosaic_sample['ctr'].append(sample['ctr'])
                    mosaic_sample['rad'].append(sample['rad'])
                    mosaic_sample['cls'].append(sample['cls'])
                mosaic_sample['ctr_transform'].append(sample['ctr_transform'])
                mosaic_sample['feat_transform'].append(sample['feat_transform'])
            
            mosaic_sample['image'] = image
            if len(mosaic_sample['ctr']) == 0:
                mosaic_sample['ctr'] = np.zeros((0, 3), dtype=np.float32)
                mosaic_sample['rad'] = np.zeros((0, 3), dtype=np.float32)
                mosaic_sample['cls'] = np.zeros((0, 1), dtype=np.float32)
            else:
                mosaic_sample['ctr'] = np.concatenate(mosaic_sample['ctr'], axis=0)
                mosaic_sample['rad'] = np.concatenate(mosaic_sample['rad'], axis=0)
                mosaic_sample['cls'] = np.concatenate(mosaic_sample['cls'], axis=0)
                
            mosaic_sample = coordToAnnot(mosaic_sample)
            mosaic_samples.append(mosaic_sample)
            
        return mosaic_samples

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
        split_images, nzhw, image_shape = self.splitcomb.split(image)
        data['split_images'] = np.ascontiguousarray(split_images)
        data['nzhw'] = nzhw
        data['image_shape'] = image_shape
        data['spacing'] = image_spacing
        data['series_name'] = series_name
        data['series_folder'] = series_folder
        return data