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

def unsupervised_loss_fn_wrapper(loss_fn):
    def unsupervised_loss_fn(args, feats, labels, device):
        cls_loss, shape_loss, offset_loss, iou_loss = loss_fn(feats, labels, device)
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
def sharpen_prob(cls_prob, t=0.7):
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
    iter_l = iter(dataloader_l)
    # num_pseudo_label = 0
    with get_progress_bar('Train', num_iters) as progress_bar:
        for sample_u in dataloader_u:         
            optimizer.zero_grad(set_to_none=True)
            
            ### Unlabeled data
            all_Cls_feats = []
            all_Shape_feats = []
            all_Offset_feats = []
            all_feats = {}
            num_transforms_of_samples = [[] for _ in range(len(sample_u['aug_0']['image']))] # shape = (bs,)
            for key, samples in sample_u.items():
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        feats = model_predict(model, samples, device) # key: 'Cls', 'Shape', 'Offset'
                else:
                    feats = model_predict(model, samples, device)
                
                del samples['image']
                all_feats[key] = feats
                
                # Transform back
                Cls_feats = feats['Cls'].detach().cpu()
                Shape_feats = feats['Shape'].detach().cpu()
                Offset_feats = feats['Offset'].detach().cpu()
                
                feat_transforms = sample_u[key]['feat_transform'] # shape = (bs,)
                
                for b_i in range(len(feat_transforms)):
                    num_transforms_of_samples[b_i].append(len(feat_transforms[b_i]))
                    for transform in reversed(feat_transforms[b_i]):
                        Cls_feats[b_i] = transform.backward(Cls_feats[b_i])
                        Shape_feats[b_i] = transform.backward(Shape_feats[b_i])
                        Offset_feats[b_i] = transform.backward(Offset_feats[b_i])

                all_Cls_feats.append(Cls_feats)
                all_Shape_feats.append(Shape_feats)
                all_Offset_feats.append(Offset_feats)            
            
            # Ensemble
            all_Cls_feats = torch.stack(all_Cls_feats, dim=1) # shape: (bs, num_aug, 1, d, h, w)
            
            ensembled_Cls_feats = all_Cls_feats.mean(dim=1) # shape: (bs, 1, d, h, w)
            ensembled_Shape_feats = []
            ensembled_Offset_feats = []
            for b_i in range(len(num_transforms_of_samples)):
                # Get less transformed features
                idx = np.argmin(num_transforms_of_samples[b_i])
                ensembled_Shape_feats.append(all_Shape_feats[idx][b_i])
                ensembled_Offset_feats.append(all_Offset_feats[idx][b_i])
            
            ensembled_Shape_feats = torch.stack(ensembled_Shape_feats, dim=0) # shape: (bs, 3, d, h, w)
            ensembled_Offset_feats = torch.stack(ensembled_Offset_feats, dim=0) # shape: (bs, 3, d, h, w)
            # Sharpening by temperature
            ensembled_Cls_probs = ensembled_Cls_feats.sigmoid()
            ensembled_Cls_probs = sharpen_prob(ensembled_Cls_probs, t=args.sharpen_temperature) # shape: (bs, 1, d, h, w) 
            ensembled_Cls_probs = torch.clamp(ensembled_Cls_probs, 1e-4, 1 - 1e-4)
            all_loss_pseu = []
            for key in all_feats.keys(): # key: 'aug_0', 'aug_1', ...
                transformed_ensembled_Cls_probs = ensembled_Cls_probs.clone()
                transformed_Shape_feats = ensembled_Shape_feats.clone()
                transformed_Offset_feats = ensembled_Offset_feats.clone()
                
                # Transform forward
                feat_transforms = sample_u[key]['feat_transform']
                for b_i in range(len(feat_transforms)):
                    for transform in feat_transforms[b_i]:
                        transformed_ensembled_Cls_probs[b_i] = transform.forward(transformed_ensembled_Cls_probs[b_i])
                        transformed_Shape_feats[b_i] = transform.forward(transformed_Shape_feats[b_i])
                        transformed_Offset_feats[b_i] = transform.forward(transformed_Offset_feats[b_i])
                
                ##TODO For Debug
                # feats = {'Cls': transformed_ensembled_Cls_probs,
                #         'Shape': transformed_Shape_feats,
                #         'Offset': transformed_Offset_feats}
                # outputs_t = detection_postprocess(feats, device=device, threshold = 0.5, is_logits = False)
                # np.save('outputs_t.npy', outputs_t.cpu().numpy())
                # np.save('image.npy', sample_u[key]['image'].numpy())
                # raise ValueError('Check transformed features')
                
                feats = all_feats[key]
                
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        labels = {'Cls': transformed_ensembled_Cls_probs.to(device, non_blocking=True, memory_format=memory_format),
                                'Shape': transformed_Shape_feats.to(device, non_blocking=True, memory_format=memory_format),
                                'Offset': transformed_Offset_feats.to(device, non_blocking=True, memory_format=memory_format)}
                        loss_pseu, cls_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss = unsupervised_loss_fn(args, feats, labels, device)
                else:
                    labels = {'Cls': transformed_ensembled_Cls_probs.to(device, non_blocking=True, memory_format=memory_format),
                                'Shape': transformed_Shape_feats.to(device, non_blocking=True, memory_format=memory_format),
                                'Offset': transformed_Offset_feats.to(device, non_blocking=True, memory_format=memory_format)}
                    loss_pseu, cls_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss = unsupervised_loss_fn(args, feats, labels, device)
                avg_pseu_cls_loss.update(cls_pseu_loss.item(), 1 / num_aug)
                avg_pseu_shape_loss.update(shape_pseu_loss.item(), 1 / num_aug)
                avg_pseu_offset_loss.update(offset_pseu_loss.item(), 1 / num_aug)
                avg_pseu_iou_loss.update(iou_pseu_loss.item(), 1 / num_aug)
                avg_pseu_loss.update(loss_pseu.item(), 1 / num_aug)
                
                all_loss_pseu.append(loss_pseu)
                del labels
            loss_pseu = torch.mean(torch.stack(all_loss_pseu))
            del all_feats
            
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
                loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, labeled_sample, device)
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
            progress_bar.set_postfix(loss_l = avg_loss.avg,
                                    cls_l = avg_cls_loss.avg,
                                    shape_l = avg_shape_loss.avg,
                                    offset_l = avg_offset_loss.avg,
                                    giou_l = avg_iou_loss.avg,
                                    loss_u = avg_pseu_loss.avg,
                                    cls_u = avg_pseu_cls_loss.avg,
                                    shape_u = avg_pseu_shape_loss.avg,
                                    offset_u = avg_pseu_offset_loss.avg,
                                    giou_u = avg_pseu_iou_loss.avg)
            progress_bar.update()
            
            del loss, loss_pseu, outputs
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
    return metrics