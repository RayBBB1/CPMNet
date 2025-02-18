import os
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.average_meter import AverageMeter
from utils.utils import get_progress_bar

logger = logging.getLogger(__name__)

def train_one_step_wrapper(memory_format):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        cls_loss, shape_loss, offset_loss, iou_loss = model([image, labels])
        cls_loss, shape_loss, offset_loss, iou_loss = cls_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * cls_loss + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        return loss, cls_loss, shape_loss, offset_loss, iou_loss
    return train_one_step

def train(args,
          model: nn.modules,
          optimizer: torch.optim.Optimizer,
          dataloader: DataLoader,
          device: torch.device,
          ema = None,) -> Dict[str, float]:
    model.train()
    avg_cls_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    iters_to_accumulate = args.iters_to_accumulate
    # mixed precision training
    mixed_precision = args.mixed_precision
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        
    total_num_steps = len(dataloader)
    if getattr(args, 'memory_format', None) is not None and args.memory_format == 'channels_last':
        logger.info('Using channels_last memory format')
        train_one_step = train_one_step_wrapper(torch.channels_last_3d)
    else:
        train_one_step = train_one_step_wrapper(None)
        
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        if iter_i % 10  == 0 and iter_i != 0:
            dataloader.dataset.change_scale()
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, sample, device)
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        else:
            loss, cls_loss, shape_loss, offset_loss, iou_loss = train_one_step(args, model, sample, device)
            loss = loss / iters_to_accumulate
            loss.backward()
        
        # Update history
        avg_cls_loss.update(cls_loss.item() * args.lambda_cls)
        avg_shape_loss.update(shape_loss.item() * args.lambda_shape)
        avg_offset_loss.update(offset_loss.item() * args.lambda_offset)
        avg_iou_loss.update(iou_loss.item() * args.lambda_iou)
        avg_loss.update(loss.item() * iters_to_accumulate)
        
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
            
            progress_bar.set_postfix(loss = avg_loss.avg,
                                    cls_Loss = avg_cls_loss.avg,
                                    shape_loss = avg_shape_loss.avg,
                                    offset_loss = avg_offset_loss.avg,
                                    giou_loss = avg_iou_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics