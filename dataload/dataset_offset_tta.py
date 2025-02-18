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