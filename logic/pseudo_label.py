import math
import logging
import numpy as np
from typing import List, Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.box_utils import nms_3D
from dataload.utils import ALL_LOC, ALL_RAD, ALL_CLS, ALL_PROB, NODULE_SIZE
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
    
def gen_pseu_labels(model: nn.Module,
                    dataloader: DataLoader,
                    device: torch.device,
                    detection_postprocess,
                    batch_size: int = 2,
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
    with get_progress_bar('Pseu-Label Generation', len(dataloader)) as pbar:
        for sample in dataloader:
            data = sample['split_images'] # (bs, num_aug, 1, crop_z, crop_y, crop_x)
            lobes = sample['split_lobes'].to(device, non_blocking=True, memory_format=memory_format)
            nzhws = sample['nzhws']
            num_splits = sample['num_splits']
            series_names = sample['series_names']
            image_shapes = sample['image_shapes']
            all_ctr_transforms = sample['ctr_transforms'] # (N, num_aug)
            all_feat_transforms = sample['feat_transforms'] # (N, num_aug)
            transform_weights = sample['transform_weights'] # (N, num_aug)
            preds = []
            transform_weights = torch.from_numpy(transform_weights).to(device, non_blocking=True)
            num_aug = data.size(1)
            for i in range(int(math.ceil(data.size(0) / batch_size))):
                end = (i + 1) * batch_size
                if end > data.size(0):
                    end = data.size(0)
                input = data[i * batch_size:end] # (bs, num_aug, 1, crop_z, crop_y, crop_x)
                input = input.view(-1, 1, *input.size()[3:]).to(device, non_blocking=True, memory_format=memory_format) # (bs * num_aug, 1, crop_z, crop_y, crop_x)
                with torch.no_grad():
                    if mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred = model(input)
                    else:
                        pred = model(input)
                # Ensemble the augmentations
                Cls_output = pred['Cls'] # (bs * num_aug, 1, 24, 24, 24)
                Shape_output = pred['Shape'] # (bs * num_aug, 3, 24, 24, 24)
                Offset_output = pred['Offset'] # (bs * num_aug, 3, 24, 24, 24)
                
                _, _, d, h, w = Cls_output.size()
                Cls_output = Cls_output.view(-1, num_aug, 1, d, h, w)
                Shape_output = Shape_output.view(-1, num_aug, 3, d, h, w)
                Offset_output = Offset_output.view(-1, num_aug, 3, d, h, w)
                
                feat_transforms = all_feat_transforms[i * batch_size:end] # (bs, num_aug)
                for b_i in range(len(feat_transforms)):
                    for aug_i in range(num_aug):
                        if len(feat_transforms[b_i][aug_i]) > 0:
                            for trans in reversed(feat_transforms[b_i][aug_i]):
                                Cls_output[b_i, aug_i, ...] = trans.backward(Cls_output[b_i, aug_i, ...])
                                Shape_output[b_i, aug_i, ...] = trans.backward(Shape_output[b_i, aug_i, ...])
                transform_weight = transform_weights[i * batch_size:end] # (bs, num_aug)
                transform_weight = transform_weight.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5) # (bs, num_aug, 1, 1, 1, 1)
                Cls_output = (Cls_output * transform_weight).sum(1) # (bs, 1, 24, 24, 24)
                Cls_output = Cls_output.sigmoid()
                ignore_offset = 2
                Cls_output[:, :, :ignore_offset, :, :] = 0
                Cls_output[:, :, :, :ignore_offset, :] = 0
                Cls_output[:, :, :, :, :ignore_offset] = 0
                Cls_output[:, :, -ignore_offset:, :, :] = 0
                Cls_output[:, :, :, -ignore_offset:, :] = 0
                Cls_output[:, :, :, :, -ignore_offset:] = 0
                
                Shape_output = (Shape_output * transform_weight).sum(1) # (bs, 3, 24, 24, 24)
                Offset_output = Offset_output[:, 0, ...] # (bs, 3, 24, 24, 24)
                pred = {'Cls': Cls_output, 'Shape': Shape_output, 'Offset': Offset_output}
                lobe = lobes[i * batch_size:end]
                
                pred = detection_postprocess(pred, device=device, is_logits=False, lobe_mask = lobe) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                preds.append(pred.data.cpu().numpy())
                del input, Cls_output, Shape_output, Offset_output, pred
            del data, lobes
            
            preds = np.concatenate(preds, 0)
            start_idx = 0
            for i in range(len(num_splits)):
                n_split = num_splits[i]
                nzhw = nzhws[i]
                image_shape = image_shapes[i]
                series_name = series_names[i]
                pred = split_comber.combine(preds[start_idx:start_idx + n_split], nzhw, image_shape)
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
                start_idx += n_split
            pbar.update(1)
    return pseudo_labels