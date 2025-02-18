import math
import logging
import numpy as np
from typing import List, Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.box_utils import nms_3D
from dataload.utils import ALL_LOC, ALL_RAD, ALL_CLS, ALL_PROB, ALL_IOU, ALL_HARD_FP_LOC, ALL_HARD_FP_RAD, ALL_HARD_FP_PROB, load_label, gen_label_path, compute_bbox3d_iou
from utils.utils import get_progress_bar
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def pred2label(pred: np.ndarray) -> Dict[str, np.ndarray]:
    pred = pred[..., 1:] # List of [prob, ctr_z, ctr_y, ctr_x, d, h, w]
        
    all_prob = []
    all_loc = []
    all_rad = []
    all_cls = []
    for i in range(len(pred)):
        prob, ctr_z, ctr_y, ctr_x, d, h, w = pred[i]
        all_loc.append([ctr_z, ctr_y, ctr_x])
        all_rad.append([d, h, w])
        all_cls.append(0)
        all_prob.append(prob)
        
    if len(all_loc) == 0:
        label = {ALL_LOC: np.zeros((0, 3)),
                ALL_RAD: np.zeros((0,)),
                ALL_CLS: np.zeros((0, 3), dtype=np.int32),
                ALL_PROB: np.zeros((0,))}
    else:
        label = {ALL_LOC: np.array(all_loc),
                ALL_RAD: np.array(all_rad),
                ALL_CLS: np.array(all_cls, dtype=np.int32),
                ALL_PROB: np.array(all_prob)}
    return label
    
def get_bboxes(label: dict):
    ctr = label[ALL_LOC]
    rad = label[ALL_RAD]
    if len(ctr) != 0:
        return np.stack([ctr - rad / 2, ctr + rad / 2], axis=1) # (N, 2, 3)
    else:
        return np.zeros((0, 2, 3))    

def gen_hard_FP(model: nn.Module,
                dataloader: DataLoader,
                device: torch.device,
                detection_postprocess,
                batch_size: int = 16,
                nms_keep_top_k: int = 40,
                mixed_precision: bool = False,
                memory_format: str = None) -> Dict[str, np.ndarray]:
    """
    Return:
        A dictionary with series name as key and pseudo labels as value. The pseudo label is a dictionary with keys 'all_loc', 'all_rad', 'all_cls'.
    """
    logger.info("Generating pseudo labels")
    
    model.eval()
    split_comber = dataloader.dataset.splitcomb
    memory_format = get_memory_format(memory_format)
    
    pseudo_labels = dict()
    all_series_names = []
    all_series_folders = []
    
    with get_progress_bar('Hard FP Generation', len(dataloader)) as pbar:
        for sample in dataloader:
            data = sample['split_images'].to(device, non_blocking=True, memory_format=memory_format)
            nzhws = sample['nzhws']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            series_folders = sample['series_folders']
            image_shapes = sample['image_shapes']
            preds = []
            for i in range(int(math.ceil(data.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > data.size(0):
                    end = data.size(0)
                input = data[i * batch_size:end]
                
                with torch.no_grad():
                    if mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred = model(input)
                            pred = detection_postprocess(pred, device=device) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                    else:
                        pred = model(input)
                        pred = detection_postprocess(pred, device=device)
                        
                preds.append(pred.data.cpu().numpy())
            
            preds = np.concatenate(preds, 0) # [n, 8]
            
            start_index = 0
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                nzhw = nzhws[i]
                series_name = series_names[i]
                series_folder = series_folders[i]
                image_shape = image_shapes[i]
                pred = split_comber.combine(preds[start_index:start_index + n_split], nzhw, image_shape)
                
                pred = torch.from_numpy(pred).view(-1, 8)
                # Remove the padding
                valid_mask = (pred[:, 0] != -1.0)
                pred = pred[valid_mask]
                # NMS
                if len(pred) > 0:
                    keep = nms_3D(pred[:, 1:], overlap=0.05, top_k=nms_keep_top_k)
                    pred = pred[keep]
                pred = pred.numpy()
                pseudo_labels[series_name] = pred2label(pred)
                start_index += n_split
                
                all_series_folders.append(series_folder)
                all_series_names.append(series_name)
            pbar.update(1)

    labels = dict()
    image_spacing = dataloader.dataset.image_spacing
    for series_name, series_folder in zip(all_series_names, all_series_folders):
        label_path = gen_label_path(series_folder, series_name)
        label = load_label(label_path, image_spacing)
        if len(label[ALL_CLS]) != 0:
            label[ALL_RAD] = label[ALL_RAD] / image_spacing
        labels[series_name] = label

    all_samples = dict()
    for series_name in pseudo_labels.keys():
        label = labels[series_name]
        pseu_label = pseudo_labels[series_name]
        pseu_ctrs = pseu_label[ALL_LOC]
        pseu_rads = pseu_label[ALL_RAD]
        pseud_prob = pseu_label[ALL_PROB]
        
        gt_bboxes = get_bboxes(label)
        pseu_bboxes = get_bboxes(pseu_label)
        
        if len(gt_bboxes) == 0:
            all_samples[series_name] = {ALL_RAD: np.zeros((0, 3)),
                                        ALL_LOC: np.zeros((0, 3)),
                                        ALL_PROB: np.zeros((0,)),
                                        ALL_IOU: np.zeros((0,)),
                                        ALL_CLS: np.zeros((0, 3), dtype=np.int32),
                                        ALL_HARD_FP_LOC: pseu_ctrs, 
                                        ALL_HARD_FP_RAD: pseu_rads,
                                        ALL_HARD_FP_PROB: pseud_prob}
            continue
        elif len(pseu_bboxes) == 0:
            all_samples[series_name] = {ALL_RAD: label[ALL_RAD],
                                        ALL_LOC: label[ALL_LOC],
                                        ALL_PROB: np.zeros((0,)),
                                        ALL_IOU: np.zeros((0,)),
                                        ALL_CLS: np.zeros((0, 3), dtype=np.int32),
                                        ALL_HARD_FP_LOC: np.zeros((0, 3)),
                                        ALL_HARD_FP_RAD: np.zeros((0,)),
                                        ALL_HARD_FP_PROB: np.zeros((0,))}
            continue
        
        ious = compute_bbox3d_iou(pseu_bboxes, gt_bboxes)
        
        # Collect positive samples
        pos_all_ctrs = []
        pos_all_rads = []
        pos_all_probs = []
        pos_all_ious = []
        gt_max_ious = ious.max(axis=0)
        gt_match_idx = ious.argmax(axis=0)
        for i, (iou, pred_idx) in enumerate(zip(gt_max_ious, gt_match_idx)):
            if iou > 1e-3:
                prob = pseud_prob[pred_idx]
            else:
                prob = 0.0
            pos_all_ctrs.append(label[ALL_LOC][i])
            pos_all_rads.append(label[ALL_RAD][i])
            pos_all_probs.append(prob)
            pos_all_ious.append(iou)
        
        all_samples[series_name] = dict()
        all_samples[series_name][ALL_RAD] = np.array(pos_all_rads)
        all_samples[series_name][ALL_LOC] = np.array(pos_all_ctrs)
        all_samples[series_name][ALL_PROB] = np.array(pos_all_probs)
        all_samples[series_name][ALL_IOU] = np.array(pos_all_ious)
        all_samples[series_name][ALL_CLS] = np.zeros((len(pos_all_ious),), dtype=np.int32)
        
        fp_mask = (ious.max(axis=1) < 1e-3)
        if np.sum(fp_mask) == 0:
            neg_all_ctrs = np.zeros((0, 3))
            neg_all_rads = np.zeros((0,))
            neg_all_probs = np.zeros((0,))
        else:
            neg_all_ctrs = pseu_ctrs[fp_mask]
            neg_all_rads = pseu_rads[fp_mask]
            neg_all_probs = pseud_prob[fp_mask]
        
        all_samples[series_name][ALL_HARD_FP_LOC] = neg_all_ctrs
        all_samples[series_name][ALL_HARD_FP_RAD] = neg_all_rads
        all_samples[series_name][ALL_HARD_FP_PROB] = neg_all_probs
    return all_samples