import os
import json
import math
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format
from transform.label import CoordToAnnot

logger = logging.getLogger(__name__)

# def unsupervised_train_one_step_wrapper(memory_format, loss_fn):
#     def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], background_mask, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
#         labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
#         # Compute loss
#         outputs = model(image)
#         cls_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, background_mask, device = device)
#         cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
#         loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
#         return loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs
#     return train_one_step

def unsupervised_loss_fn_wrapper(loss_fn):
    def unsupervised_loss_fn(args, feats, pseudo_labels, background_mask, device):
        cls_loss, shape_loss, offset_loss, iou_loss = loss_fn(feats, pseudo_labels, background_mask, device)
        loss = args.lambda_pseu_cls * cls_loss + args.lambda_pseu_shape * shape_loss + args.lambda_pseu_offset * offset_loss + args.lambda_pseu_iou * iou_loss
        return loss, cls_loss, shape_loss, offset_loss, iou_loss
    return unsupervised_loss_fn

def train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        outputs = model(image)
        cls_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, device = device)
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        return loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

def model_predict_wrapper(memory_format):
    def model_predict(model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        outputs = model(image) # Dict[str, torch.Tensor], key: 'Cls', 'Shape', 'Offset'
        return outputs
    return model_predict

@torch.no_grad()
def sharpen_prob(cls_prob, t=0.5):
    cls_prob_s = cls_prob ** (1 / t)
    return (cls_prob_s / (cls_prob_s + (1 - cls_prob_s) ** (1 / t)))

def train(args,
          model: nn.modules,
          detection_loss,
          unsupervised_detection_loss,
          optimizer: torch.optim.Optimizer,
          dataloader_u: DataLoader,
          dataloader_l: DataLoader,
          detection_postprocess,
          num_iters: int,
          device: torch.device) -> Dict[str, float]:
    model.train()
    
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    avg_pseu_cls_loss = AverageMeter()
    avg_pseu_shape_loss = AverageMeter()
    avg_pseu_offset_loss = AverageMeter()
    avg_pseu_iou_loss = AverageMeter()
    avg_pseu_loss = AverageMeter()
    
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format, detection_loss)
    model_predict = model_predict_wrapper(memory_format)
    unsupervised_loss_fn = unsupervised_loss_fn_wrapper(unsupervised_detection_loss)
    
    num_aug = args.num_aug
    optimizer.zero_grad()
    iter_l = iter(dataloader_l)
    
    num_pseudo_label = 0
    coord_to_annot = CoordToAnnot()
    with get_progress_bar('Train', num_iters) as progress_bar:
        for sample_u in dataloader_u:         
            optimizer.zero_grad(set_to_none=True)
            
            ### Unlabeled data
            all_Cls_feats = []
            all_Shape_feats = []
            all_Offset_feats = []
            all_feats = {}
            for key, samples in sample_u.items():
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        feats = model_predict(model, samples, device) # key: 'Cls', 'Shape', 'Offset'
                else:
                    feats = model_predict(model, samples, device)
                
                all_feats[key] = feats
                
                # Transform back
                Cls_feats = feats['Cls'].detach()
                Shape_feats = feats['Shape'].detach()
                Offset_feats = feats['Offset'].detach()
                
                feat_transforms = sample_u[key]['feat_transform'] # shape = (bs,)
                for b_i in range(len(feat_transforms)):
                    for transform in reversed(feat_transforms[b_i]):
                        Cls_feats[b_i] = transform.backward(Cls_feats[b_i])
                        Shape_feats[b_i] = transform.backward(Shape_feats[b_i])
                        Offset_feats[b_i] = transform.backward(Offset_feats[b_i])
                
                all_Cls_feats.append(Cls_feats)
                all_Shape_feats.append(Shape_feats)
                all_Offset_feats.append(Offset_feats)            
            
            # Ensemble
            all_Cls_feats = torch.stack(all_Cls_feats, dim=1) # shape: (bs, num_aug, 1, d, h, w)
            all_Shape_feats = torch.stack(all_Shape_feats, dim=1) # shape: (bs, num_aug, 3, d, h, w)
            all_Offset_feats = torch.stack(all_Offset_feats, dim=1) # shape: (bs, num_aug, 3, d, h, w)
            
            all_Cls_feats = all_Cls_feats.mean(dim=1) # shape: (bs, 1, d, h, w)
            # arg_max_Cls = all_Cls_feats.argmax(dim=1, keepdim=True) # shape: (bs, 1, d, h, w)
            all_Shape_feats = all_Shape_feats.mean(dim=1) # shape: (bs, 3, d, h, w)
            all_Offset_feats = all_Offset_feats.mean(dim=1) # shape: (bs, 3, d, h, w)
            
            # all_Shape_feats = all_Shape_feats.gather(1, arg_max_Cls.repeat(1, 1, 3, 1, 1, 1)) # shape: (bs, 3, d, h, w)
            # all_Offset_feats = all_Offset_feats.gather(1, arg_max_Cls.repeat(1, 1, 3, 1, 1, 1)) # shape: (bs, 3, d, h, w)
            
            # Sharpening by temperature
            all_Cls_probs = all_Cls_feats.sigmoid()
            all_Cls_probs = sharpen_prob(all_Cls_probs, t=args.sharpen_temperature) # shape: (bs, 1, d, h, w) 
            
            feat_transforms = sample_u[key]['feat_transform'] # shape = (bs,)
            
            Cls_probs = all_Cls_probs.clone()
            Shape_feats = all_Shape_feats.clone()
            Offset_feats = all_Offset_feats.clone()
            feats = {'Cls': Cls_probs, 'Shape': Shape_feats, 'Offset': Offset_feats}
            
            # shape: (bs, top_k, 8)
            # => top_k (default = 60) 
            # => 8: 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
            outputs_t = detection_postprocess(feats, device=device, threshold = args.pseudo_label_threshold, is_logits=False)
            valid_mask = (outputs_t[..., 0] != -1.0)
            outputs_t = outputs_t.cpu().numpy()
            
            all_loss_pseu = []
            num_cur_pseudo_label = 0
            if torch.count_nonzero(valid_mask) != 0:
                bs = outputs_t.shape[0]
                # Calculate background mask
                for key in all_feats.keys():
                    background_mask = (Cls_probs < (1 - args.pseudo_background_threshold)) # shape: (bs, 1, d, h, w)
                    # Transform forward
                    feat_transforms = sample_u[key]['feat_transform']
                    for transform in feat_transforms[b_i]:
                        background_mask[b_i] = transform.forward(background_mask[b_i])
                
                    ctr_transforms = sample_u[key]['ctr_transform']
                    
                    spacings = np.array([[1.0, 0.8, 0.8]]).repeat(bs, axis=0)
                    transformed_annots = []
                    for b_i in range(bs):
                        output = outputs_t[b_i].copy()
                        valid_mask = (output[:, 0] != -1.0)
                        output = output[valid_mask]
                        if len(output) == 0:
                            transformed_annots.append(np.zeros((0, 10), dtype='float32'))
                            continue
                        ctrs = output[:, 2:5]
                        shapes = output[:, 5:8]
                        spacing = spacings[b_i]
                        for transform in ctr_transforms[b_i]:
                            ctrs = transform.forward_ctr(ctrs)
                            shapes = transform.forward_rad(shapes)
                            spacing = transform.forward_spacing(spacing)
                        
                        sample = {'ctr': ctrs, 
                                'rad': shapes, 
                                'cls': np.zeros((len(ctrs), 1), dtype='int32'),
                                'spacing': spacing}
                        
                        sample = coord_to_annot(sample)
                        transformed_annots.append(sample['annot'])
            
                    valid_mask = np.array([len(annot) > 0 for annot in transformed_annots], dtype=np.int32)
                    valid_mask = (valid_mask == 1)
                    max_num_annots = max(annot.shape[0] for annot in transformed_annots)
                    if max_num_annots > 0:
                        transformed_annots_padded = np.ones((len(transformed_annots), max_num_annots, 10), dtype='float32') * -1
                        for idx, annot in enumerate(transformed_annots):
                            if annot.shape[0] > 0:
                                transformed_annots_padded[idx, :annot.shape[0], :] = annot
                    else:
                        transformed_annots_padded = np.ones((len(transformed_annots), 1, 10), dtype='float32') * -1

                    transformed_annots_padded = transformed_annots_padded[valid_mask]
                    transformed_annots_padded = torch.from_numpy(transformed_annots_padded)
                    
                    background_mask = background_mask.view(bs, -1) # shape: (bs, num_points)
                    background_mask = background_mask[valid_mask]
                    
                    # print('background_mask', background_mask.shape)
                    # print('transformed_annots_padded', transformed_annots_padded.shape)
                    
                    # print(transformed_annots_padded)
                    valid_mask = torch.from_numpy(valid_mask).to(device, non_blocking=True)
                    transformed_annots_padded = transformed_annots_padded.to(device, non_blocking=True)
                    valid_feats = dict()
                    for key, value in all_feats[key].items():
                        valid_feats[key] = value[valid_mask]
                    
                    loss_pseu, cls_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss = unsupervised_loss_fn(args, valid_feats, transformed_annots_padded,background_mask, device)
                
                    avg_pseu_cls_loss.update(cls_pseu_loss.item() * args.lambda_pseu_cls)
                    avg_pseu_shape_loss.update(shape_pseu_loss.item() * args.lambda_pseu_shape)
                    avg_pseu_offset_loss.update(offset_pseu_loss.item() * args.lambda_pseu_offset)
                    avg_pseu_iou_loss.update(iou_pseu_loss.item() * args.lambda_pseu_iou)
                    avg_pseu_loss.update(loss_pseu.item())
                    all_loss_pseu.append(loss_pseu)
                    print('loss_pseu', loss_pseu)
                    num_cur_pseudo_label += len(transformed_annots_padded)
                    del transformed_annots_padded, background_mask
                    
            if len(all_loss_pseu) > 0:
                num_pseudo_label += (num_cur_pseudo_label / num_aug)
                loss_pseu = torch.mean(torch.stack(all_loss_pseu))
            else:
                loss_pseu = torch.tensor(0.0, device=device)
            del all_feats, all_Cls_feats, all_Shape_feats, all_Offset_feats
            ### Labeled data
            try:
                labeled_sample = next(iter_l)
            except StopIteration:
                iter_l = iter(dataloader_l)
                labeled_sample = next(iter_l)
            
            print("start to forward labeled data")
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, labeled_sample, device)
            else:
                loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, labeled_sample, device)
            avg_cls_loss.update(cls_loss.item() * args.lambda_cls)
            avg_shape_loss.update(shape_loss.item() * args.lambda_shape)
            avg_offset_loss.update(offset_loss.item() * args.lambda_offset)
            avg_iou_loss.update(iou_loss.item() * args.lambda_iou)
            avg_loss.update(loss.item())
            print("end of forward labeled data")
            # Update model
            total_loss = loss + loss_pseu * args.lambda_pseu
            if mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            print("end of backward")
            progress_bar.set_postfix(loss_l = avg_loss.avg,
                                    cls_l = avg_cls_loss.avg,
                                    shape_l = avg_shape_loss.avg,
                                    offset_l = avg_offset_loss.avg,
                                    giou_l = avg_iou_loss.avg,
                                    loss_u = avg_pseu_loss.avg,
                                    cls_u = avg_pseu_cls_loss.avg,
                                    shape_u = avg_pseu_shape_loss.avg,
                                    offset_u = avg_pseu_offset_loss.avg,
                                    giou_u = avg_pseu_iou_loss.avg,
                                    num_u = num_pseudo_label)
            progress_bar.update()
            
            del loss, loss_pseu, outputs
            torch.cuda.empty_cache()
            ##TODO update BN?
    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg,
                'loss_pseu': avg_pseu_loss.avg,
                'cls_loss_pseu': avg_pseu_cls_loss.avg,
                'shape_loss_pseu': avg_pseu_shape_loss.avg,
                'offset_loss_pseu': avg_pseu_offset_loss.avg,
                'iou_loss_pseu': avg_pseu_iou_loss.avg}
    logger.info(f'Num pseudo label: {num_pseudo_label}')
    return metrics