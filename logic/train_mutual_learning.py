import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar
from .utils import get_memory_format

logger = logging.getLogger(__name__)

def train_one_step_wrapper(memory_format):
    def train_one_step(args, model: nn.modules, image, labels, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute loss
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, feats = model([image, labels])
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        # del image, labels
        return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, feats
    return train_one_step

def train(args,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          model2: nn.modules,
          dataloader: DataLoader,
          device: torch.device,
          ema = None,
          ema2 = None) -> Dict[str, float]:
    model.train()
    model2.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    avg_feat_loss = AverageMeter()
    
    avg_cls_pos_loss2 = AverageMeter()
    avg_cls_neg_loss2 = AverageMeter()
    avg_cls_loss2 = AverageMeter()
    avg_shape_loss2 = AverageMeter()
    avg_offset_loss2 = AverageMeter()
    avg_iou_loss2 = AverageMeter()
    
    iters_to_accumulate = args.iters_to_accumulate
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    total_num_steps = len(dataloader)
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format)
        
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    feat_loss_fn = nn.CosineEmbeddingLoss()
    for iter_i, sample in enumerate(dataloader):
        if mixed_precision:
            with torch.cuda.amp.autocast():
                image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
                labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, feats = train_one_step(args, model, image, labels, device)
                loss2, cls_pos_loss2, cls_neg_loss2, shape_loss2, offset_loss2, iou_loss2, feats2 = train_one_step(args, model2, image, labels, device)

                # shape of feats: [bs, c, z, y, x]
                feat_loss = 0
                for i in range(len(feats)):
                    bs, c, z, y, x = feats[i].shape
                    num_target = bs * z * y * x
                    
                    f1 = feats[i].contiguous().view(bs, c, -1).permute(0, 2, 1).contiguous().view(-1, c)
                    f2 = feats2[i].contiguous().view(bs, c, -1).permute(0, 2, 1).contiguous().view(-1, c)
                    feat_loss += feat_loss_fn(f1, f2, torch.ones(num_target, device=device)) 
            
            loss = (loss + loss2 + feat_loss * args.lambda_feat) / iters_to_accumulate    
            scaler.scale(loss).backward()
            
            del image, labels, feats, feats2
        
        # Update history
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
        
        avg_cls_pos_loss2.update(cls_pos_loss2.item())
        avg_cls_neg_loss2.update(cls_neg_loss2.item())
        avg_cls_loss2.update(cls_pos_loss2.item() + cls_neg_loss2.item())
        avg_shape_loss2.update(shape_loss2.item())
        avg_offset_loss2.update(offset_loss2.item())
        avg_iou_loss2.update(iou_loss2.item())
        
        avg_loss.update(loss.item() * iters_to_accumulate)
        avg_feat_loss.update(feat_loss.item())
        
        # Update model
        if (iter_i + 1) % iters_to_accumulate == 0 or iter_i == total_num_steps - 1:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
            
            # Update EMA
            if ema is not None:
                ema.update()
            if ema2 is not None:
                ema2.update()
            
            progress_bar.set_postfix(loss = avg_loss.avg,
                                    pos_cls = avg_cls_pos_loss.avg,
                                    neg_cls = avg_cls_neg_loss.avg,
                                    iou_loss = avg_iou_loss.avg,
                                    pos_cls2 = avg_cls_pos_loss2.avg,
                                    neg_cls2 = avg_cls_neg_loss2.avg,
                                    iou_loss2 = avg_iou_loss2.avg,
                                    feat_loss = avg_feat_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg,
                'cls_loss2': avg_cls_loss2.avg,
                'cls_pos_loss2': avg_cls_pos_loss2.avg,
                'cls_neg_loss2': avg_cls_neg_loss2.avg,
                'shape_loss2': avg_shape_loss2.avg,
                'offset_loss2': avg_offset_loss2.avg,
                'iou_loss2': avg_iou_loss2.avg,
                'feat_loss': avg_feat_loss.avg,}
    
    return metrics