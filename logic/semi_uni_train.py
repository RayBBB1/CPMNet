import os
import json
import math
import logging
import numpy as np
from typing import Any, Dict, List, Tuple

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format
from transform.label import CoordToAnnot
from dataload.utils import compute_bbox3d_iou

logger = logging.getLogger(__name__)

def unsupervised_train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], background_mask, device: torch.device, drop=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute loss
        image = sample['image']
        labels = sample['annot']
        outputs = model(image, drop)
        cls_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, background_mask, device = device)
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_pseu_cls * cls_loss + args.lambda_pseu_shape * shape_loss + args.lambda_pseu_offset * offset_loss + args.lambda_pseu_iou * iou_loss
        return loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

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
    
    # For analysis
    avg_iou_pseu = AverageMeter()
    avg_tp_pseu = AverageMeter()
    avg_fp_pseu = AverageMeter()
    avg_fn_pseu = AverageMeter()
    
    iters_to_accumulate = args.iters_to_accumulate
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format, detection_loss)
    unsupervised_train_one_step = unsupervised_train_one_step_wrapper(memory_format, unsupervised_detection_loss)
    model_predict = model_predict_wrapper(memory_format)
    
    iter_l = iter(dataloader_l)
    num_pseudo_label = 0
    coord_to_annot = CoordToAnnot()
    with get_progress_bar('Train', num_iters) as progress_bar:
        for sample_u in dataloader_u:         
            optimizer.zero_grad()
            ### Unlabeled data
            weak_u_sample = sample_u['weak']
            strong_u_sample = sample_u['strong']
            model.eval()
            with torch.no_grad():
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        feats_t = model_predict(model, weak_u_sample, device) # shape: (bs, top_k, 7)
                    del weak_u_sample['image']
                else:
                    feats_t = model_predict(model, strong_u_sample, device)
                # shape: (bs, top_k, 8)
                # => top_k (default = 60) 
                # => 8: 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                outputs_t = detection_postprocess(feats_t, device=device, threshold = args.pseudo_label_threshold) 
            model.train()
            # np.save('weak_image.npy', weak_u_sample['image'].numpy())
            # np.save('outputs_t.npy', outputs_t.cpu().numpy())
            
            # Add label
            # Remove the padding, -1 means invalid
            valid_mask = (outputs_t[..., 0] != -1.0)
            if torch.count_nonzero(valid_mask) != 0:
                bs = outputs_t.shape[0]
                # Calculate background mask
                cls_prob = feats_t['Cls'].sigmoid() # shape: (bs, 1, d, h, w)
                background_mask = (cls_prob < (1 - args.pseudo_background_threshold)) # shape: (bs, 1, d, h, w)
                
                weak_feat_transforms = weak_u_sample['feat_transform'] # shape = (bs,)
                strong_feat_transforms = strong_u_sample['feat_transform'] # shape = (bs,)
                for b_i in range(bs):
                    for transform in reversed(weak_feat_transforms[b_i]):
                        background_mask[b_i] = transform.backward(background_mask[b_i])
                    for transform in strong_feat_transforms[b_i]:
                        background_mask[b_i] = transform.forward(background_mask[b_i])
                
                outputs_t = outputs_t.cpu().numpy()
                weak_ctr_transforms = weak_u_sample['ctr_transform'] # shape = (bs,)
                strong_ctr_transforms = strong_u_sample['ctr_transform'] # shape = (bs,)
                weak_spacings = weak_u_sample['spacing'] # shape = (bs,)
                
                transformed_annots = []
                for b_i in range(bs):
                    output = outputs_t[b_i]
                    valid_mask = (output[:, 0] != -1.0)
                    output = output[valid_mask]
                    if len(output) == 0:
                        transformed_annots.append(np.zeros((0, 10), dtype='float32'))
                        continue
                    ctrs = output[:, 2:5]
                    shapes = output[:, 5:8]
                    spacing = weak_spacings[b_i]
                    for transform in reversed(weak_ctr_transforms[b_i]):
                        ctrs = transform.backward_ctr(ctrs)
                        shapes = transform.backward_rad(shapes)
                        spacing = transform.backward_spacing(spacing)
                    
                    for transform in strong_ctr_transforms[b_i]:
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

                # (For analysis) Compute iou between pseudo label and original label
                all_iou_pseu = []
                tp, fp, fn = 0, 0, 0
                for annot, pseudo_annot, is_valid in zip(strong_u_sample['annot'].numpy(), transformed_annots_padded, valid_mask):
                    if not is_valid:
                        fn += np.count_nonzero((annot[:, -1] != -1))
                        continue
                    
                    annot = annot[annot[:, -1] != -1] # (ctr_z, ctr_y, ctr_x, d, h, w, space_z, space_y, space_x)
                    pseudo_annot = pseudo_annot[pseudo_annot[:, -1] != -1]
                    
                    if len(annot) == 0:
                        fp += len(pseudo_annot)
                        continue
                    elif len(pseudo_annot) == 0:
                        fn += len(annot)
                        continue
                    
                    bboxes = np.stack([annot[:, :3] - annot[:, 3:6] / 2, annot[:, :3] + annot[:, 3:6] / 2], axis=1)
                    pseudo_bboxes = np.stack([pseudo_annot[:, :3] - pseudo_annot[:, 3:6] / 2, pseudo_annot[:, :3] + pseudo_annot[:, 3:6] / 2], axis=1)
                    ious = compute_bbox3d_iou(pseudo_bboxes, bboxes)
                    
                    iou_pseu = ious.max(axis=1)
                    iou = ious.max(axis=0)
                    
                    all_iou_pseu.extend(iou_pseu.tolist())    
                    tp += np.count_nonzero(iou > 1e-3)
                    fp += np.count_nonzero(iou_pseu < 1e-3)
                
                avg_iou_pseu.update(np.mean(all_iou_pseu))
                avg_tp_pseu.update(tp)
                avg_fp_pseu.update(fp)
                avg_fn_pseu.update(fn)
                

                transformed_annots_padded = transformed_annots_padded[valid_mask]
                
                strong_u_sample['image'] = strong_u_sample['image'][valid_mask]
                strong_u_sample['annot'] = torch.from_numpy(transformed_annots_padded)
                
                background_mask = background_mask.view(bs, -1) # shape: (bs, num_points)
                background_mask = background_mask[valid_mask]
                
                # np.save('image.npy', strong_u_sample['image'].numpy())
                # np.save('annot.npy', strong_u_sample['annot'].numpy())
                # np.save('background_mask.npy', background_mask.cpu().numpy())
                # raise ValueError('Check the pseudo label')
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        strong_u_sample['image'] = strong_u_sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
                        strong_u_sample['annot'] = strong_u_sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
                        loss_pseu, cls_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss, outputs_pseu = unsupervised_train_one_step(args, model, strong_u_sample, background_mask, device)
                        loss_pseu2, cls_pseu_loss2, shape_pseu_loss2, offset_pseu_loss2, iou_pseu_loss2, outputs_pseu2 = unsupervised_train_one_step(args, model, strong_u_sample, background_mask, device, drop=0.5)
                        
                        loss_pseu = (loss_pseu + loss_pseu2) / 2
                        cls_pseu_loss = (cls_pseu_loss + cls_pseu_loss2) / 2
                        shape_pseu_loss = (shape_pseu_loss + shape_pseu_loss2) / 2
                        offset_pseu_loss = (offset_pseu_loss + offset_pseu_loss2) / 2
                        iou_pseu_loss = (iou_pseu_loss + iou_pseu_loss2) / 2

                    
                avg_pseu_cls_loss.update(cls_pseu_loss.item())
                avg_pseu_shape_loss.update(shape_pseu_loss.item())
                avg_pseu_offset_loss.update(offset_pseu_loss.item())
                avg_pseu_iou_loss.update(iou_pseu_loss.item())
                avg_pseu_loss.update(loss_pseu.item())
                
                num_pseudo_label += len(strong_u_sample['annot'])
                del outputs_pseu, outputs_pseu2, strong_u_sample, background_mask
            else:
                loss_pseu = torch.tensor(0.0, device=device)
                for annot in zip(strong_u_sample['annot'].numpy()):
                    if len(annot) > 0:
                        avg_fn_pseu.update(np.count_nonzero(annot[annot[:, -1] != -1]))
                del outputs_t
                
            del feats_t, weak_u_sample
            ### Labeled data
            try:
                labeled_sample = next(iter_l)
            except StopIteration:
                iter_l = iter(dataloader_l)
                labeled_sample = next(iter_l)
            
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, labeled_sample, device)
            else:
                loss, cls_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, labeled_sample, device)
                
            avg_cls_loss.update(cls_loss.item())
            avg_shape_loss.update(shape_loss.item())
            avg_offset_loss.update(offset_loss.item())
            avg_iou_loss.update(iou_loss.item())
            avg_loss.update(loss.item())
            
            # Update model
            total_loss = loss + loss_pseu * args.lambda_pseu
            if mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            del labeled_sample, outputs, loss, loss_pseu
            progress_bar.set_postfix(loss_l = avg_loss.avg,
                                    cls_l = avg_cls_loss.avg,
                                    giou_l = avg_iou_loss.avg,
                                    loss_u = avg_pseu_loss.avg,
                                    cls_u = avg_pseu_cls_loss.avg,
                                    giou_u = avg_pseu_iou_loss.avg,
                                    num_u = num_pseudo_label,
                                    tp = avg_tp_pseu.sum,
                                    fp = avg_fp_pseu.sum,
                                    fn = avg_fn_pseu.sum)
            # progress_bar.set_postfix(loss_l = avg_loss.avg,
            #                         cls_l = avg_cls_loss.avg,
            #                         shape_l = avg_shape_loss.avg,
            #                         offset_l = avg_offset_loss.avg,
            #                         giou_l = avg_iou_loss.avg,
            #                         loss_u = avg_pseu_loss.avg,
            #                         cls_u = avg_pseu_cls_loss.avg,
            #                         shape_u = avg_pseu_shape_loss.avg,
            #                         offset_u = avg_pseu_offset_loss.avg,
            #                         giou_u = avg_pseu_iou_loss.avg,
            #                         num_u = num_pseudo_label)
            progress_bar.update()
            torch.cuda.empty_cache()
            
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
    # For analysis
    logger.info(f'Num pseudo label: {num_pseudo_label}')
    recall = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fn_pseu.sum, 1e-3)
    precision = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fp_pseu.sum, 1e-3)
    logger.info('Pseudo Recall: {:.3f}'.format(recall))
    logger.info('Pseudo Precision: {:.3f}'.format(precision))
    logger.info('Pseudo TP: {}'.format(avg_tp_pseu.sum))
    logger.info('Pseudo FP: {}'.format(avg_fp_pseu.sum))
    logger.info('Pseudo FN: {}'.format(avg_fn_pseu.sum))
    return metrics