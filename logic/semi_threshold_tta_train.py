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
from dataload.utils import compute_bbox3d_iou

logger = logging.getLogger(__name__)

TTA_BATCH_SIZE = 8

def unsupervised_train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], background_mask, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        outputs = model(image)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, background_mask, device = device)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

def train_one_step_wrapper(memory_format, loss_fn):
    def train_one_step(args, model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
        labels = sample['annot'].to(device, non_blocking=True) # z, y, x, d, h, w, type[-1, 0]
        # Compute loss
        outputs = model(image)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = loss_fn(outputs, labels, device = device)
        cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = cls_pos_loss.mean(), cls_neg_loss.mean(), shape_loss.mean(), offset_loss.mean(), iou_loss.mean()
        loss = args.lambda_cls * (cls_pos_loss + cls_neg_loss) + args.lambda_shape * shape_loss + args.lambda_offset * offset_loss + args.lambda_iou * iou_loss
        return loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs
    return train_one_step

# def model_predict_wrapper(memory_format):
#     def model_predict(model: nn.modules, sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
#         image = sample['image'].to(device, non_blocking=True, memory_format=memory_format) # z, y, x
#         outputs = model(image) # Dict[str, torch.Tensor], key: 'Cls', 'Shape', 'Offset'
#         return outputs
#     return model_predict

def burn_in_train(args,
                model: nn.modules,
                detection_loss,
                optimizer: torch.optim.Optimizer,
                dataloader: DataLoader,
                device: torch.device,
                ema = None,) -> Dict[str, float]:
    model.train()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
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
    
    memory_format = get_memory_format(getattr(args, 'memory_format', None))
    if memory_format == torch.channels_last_3d:
        logger.info('Use memory format: channels_last_3d to train')
    train_one_step = train_one_step_wrapper(memory_format, detection_loss)
        
    optimizer.zero_grad()
    progress_bar = get_progress_bar('Train', (total_num_steps - 1) // iters_to_accumulate + 1)
    for iter_i, sample in enumerate(dataloader):
        if mixed_precision:
            with torch.cuda.amp.autocast():
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, sample, device)
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
        else:
            loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model, sample, device)
            loss = loss / iters_to_accumulate
            loss.backward()
        
        # Update history
        avg_cls_pos_loss.update(cls_pos_loss.item())
        avg_cls_neg_loss.update(cls_neg_loss.item())
        avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
        avg_shape_loss.update(shape_loss.item())
        avg_offset_loss.update(offset_loss.item())
        avg_iou_loss.update(iou_loss.item())
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
                                    pos_cls = avg_cls_pos_loss.avg,
                                    neg_cls = avg_cls_neg_loss.avg,
                                    cls_loss = avg_cls_loss.avg,
                                    shape_loss = avg_shape_loss.avg,
                                    offset_loss = avg_offset_loss.avg,
                                    iou_loss = avg_iou_loss.avg)
            progress_bar.update()
    
    progress_bar.close()

    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg}
    return metrics

def train(args,
          model_t: nn.modules,
          model_s: nn.modules,
          detection_loss,
          unsupervised_detection_loss,
          optimizer: torch.optim.Optimizer,
          dataloader_u: DataLoader,
          dataloader_l: DataLoader,
          detection_postprocess,
          num_iters: int,
          device: torch.device) -> Dict[str, float]:
    model_t.train()
    model_s.train()
    
    avg_cls_loss = AverageMeter()
    avg_cls_pos_loss = AverageMeter()
    avg_cls_neg_loss = AverageMeter()
    avg_shape_loss = AverageMeter()
    avg_offset_loss = AverageMeter()
    avg_iou_loss = AverageMeter()
    avg_loss = AverageMeter()
    
    avg_pseu_cls_loss = AverageMeter()
    avg_pseu_cls_pos_loss = AverageMeter()
    avg_pseu_cls_neg_loss = AverageMeter()
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
    # model_predict =model_predict_wrapper(memory_format)
    
    optimizer.zero_grad()
    iter_l = iter(dataloader_l)
    
    num_pseudo_nodules = 0
    coord_to_annot = CoordToAnnot()
    
    with get_progress_bar('Train', num_iters) as progress_bar:
        for sample_u in dataloader_u:         
            optimizer.zero_grad(set_to_none=True)
            
            ### Unlabeled data
            weak_u_sample = sample_u['weak']
            strong_u_sample = sample_u['strong']
            
            # TTA generate pseudo label
            weak_transform_weights = weak_u_sample['transform_weights'] # (N, num_aug)
            weak_transform_weights = torch.from_numpy(weak_transform_weights).to(device, non_blocking=True)
            weak_feat_transforms = weak_u_sample['feat_transform'] # (N, num_aug)
            num_aug = weak_u_sample['image'].size(1)
            
            weak_images = weak_u_sample['image']
            weak_lobes = weak_u_sample['lobe'].to(device, non_blocking=True, memory_format=memory_format)
            cls_prob = []
            outputs_t = []
            for i in range(int(math.ceil(weak_images.size(0) / TTA_BATCH_SIZE))):
                end = (i + 1) * TTA_BATCH_SIZE
                if end > weak_images.size(0):
                    end = weak_images.size(0)
                input = weak_images[i * TTA_BATCH_SIZE:end] # (bs, num_aug, 1, crop_z, crop_y, crop_x)
                input = input.view(-1, 1, *input.size()[3:]).to(device, non_blocking=True, memory_format=memory_format) # (bs * num_aug, 1, crop_z, crop_y, crop_x)
                
                if args.val_mixed_precision:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output = model_t(input)
                else:
                    with torch.no_grad():
                        output = model_t(input)

                # Ensemble the augmentations
                Cls_output = output['Cls'] # (bs * num_aug, 1, 24, 24, 24)
                Shape_output = output['Shape'] # (bs * num_aug, 3, 24, 24, 24)
                Offset_output = output['Offset'] # (bs * num_aug, 3, 24, 24, 24)
                
                _, _, d, h, w = Cls_output.size()
                Cls_output = Cls_output.view(-1, num_aug, 1, d, h, w)
                Shape_output = Shape_output.view(-1, num_aug, 3, d, h, w)
                Offset_output = Offset_output.view(-1, num_aug, 3, d, h, w)
                
                feat_transforms = weak_feat_transforms[i * TTA_BATCH_SIZE:end] # (bs, num_aug)
                for b_i in range(len(feat_transforms)):
                    for aug_i in range(num_aug):
                        if len(feat_transforms[b_i][aug_i]) > 0:
                            for trans in reversed(feat_transforms[b_i][aug_i]):
                                Cls_output[b_i, aug_i, ...] = trans.backward(Cls_output[b_i, aug_i, ...])
                                Shape_output[b_i, aug_i, ...] = trans.backward(Shape_output[b_i, aug_i, ...])
                transform_weight = weak_transform_weights[i * TTA_BATCH_SIZE:end] # (bs, num_aug)
                transform_weight = transform_weight.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5) # (bs, num_aug, 1, 1, 1, 1)
                Cls_output = (Cls_output * transform_weight).sum(1) # (bs, 1, 24, 24, 24)
                Cls_output = Cls_output.sigmoid()
                ignore_offset = 1
                Cls_output[:, :, :ignore_offset, :, :] = 0
                Cls_output[:, :, :, :ignore_offset, :] = 0
                Cls_output[:, :, :, :, :ignore_offset] = 0
                Cls_output[:, :, -ignore_offset:, :, :] = 0
                Cls_output[:, :, :, -ignore_offset:, :] = 0
                Cls_output[:, :, :, :, -ignore_offset:] = 0
                
                Shape_output = (Shape_output * transform_weight).sum(1) # (bs, 3, 24, 24, 24)
                Offset_output = Offset_output[:, 0, ...] # (bs, 3, 24, 24, 24)
                lobe = weak_lobes[i * TTA_BATCH_SIZE:end]
                output = {'Cls': Cls_output, 'Shape': Shape_output, 'Offset': Offset_output}
                
                output = detection_postprocess(output, device=device, is_logits=False, lobe_mask = lobe, threshold = args.pseudo_label_threshold, nms_topk=args.pseudo_nms_topk) #1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                outputs_t.append(output.data.cpu().numpy())
                cls_prob.append(Cls_output)
                del input, Shape_output, Offset_output, output
            del weak_lobes
            
            # Add label
            # Remove the padding, -1 means invalid
            outputs_t = np.concatenate(outputs_t, axis=0) # (bs, topk, 8)
            
            gt_ctrs = []
            gt_rads = []
            for annot in weak_u_sample['annot']:
                ctrs = []
                rads = []
                for a in annot:
                    ctrs.append(a['gt_ctr'])
                    rads.append(a['gt_rad'])
                gt_ctrs.append(ctrs)
                gt_rads.append(rads)
                
            # gt_ctrs = np.concatenate(gt_ctrs, axis=0)
            # gt_rads = np.concatenate(gt_rads, axis=0)
            # np.save('gt_ctrs.npy', gt_ctrs)
            # np.save('gt_rads.npy', gt_rads)
            # import pickle
            # with open('gt_ctrs.pkl', 'wb') as f:
            #     pickle.dump(gt_ctrs, f)
                
            # with open('gt_rads.pkl', 'wb') as f:
            #     pickle.dump(gt_rads, f)
                
            # np.save('outputs_t.npy', outputs_t)
            # np.save('weak_u_sample.npy', weak_u_sample['image'].numpy())
            
            bs = outputs_t.shape[0]
            # Calculate and transofmr background mask
            cls_prob = torch.cat(cls_prob, dim=0) # shape: (bs, 1, d, h, w)
            background_mask = (cls_prob < args.pseudo_background_threshold) # shape: (bs, 1, d, h, w)
            strong_feat_transforms = strong_u_sample['feat_transform'] # shape = (bs,)
            for batch_i in range(bs):
                for transform in strong_feat_transforms[batch_i]:
                    background_mask[batch_i] = transform.forward(background_mask[batch_i])
            # Process pseudo label
            strong_ctr_transforms = strong_u_sample['ctr_transform'] # shape = (bs,)
            weak_spacings = weak_u_sample['spacing'] # shape = (bs,)
            transformed_annots = []
            for batch_i in range(bs):
                output = outputs_t[batch_i]
                valid_mask = (output[:, 0] != -1.0)
                output = output[valid_mask]
                if len(output) == 0:
                    transformed_annots.append(np.zeros((0, 10), dtype='float32'))
                    continue
                ctrs = output[:, 2:5]
                shapes = output[:, 5:8]
                spacing = weak_spacings[batch_i]
                
                for transform in strong_ctr_transforms[batch_i]:
                    ctrs = transform.forward_ctr(ctrs)
                    shapes = transform.forward_rad(shapes)
                    spacing = transform.forward_spacing(spacing)
                
                sample = {'ctr': ctrs, 
                        'rad': shapes, 
                        'cls': np.zeros((len(ctrs), 1), dtype='int32'),
                        'spacing': spacing}
                
                sample = coord_to_annot(sample)
                transformed_annots.append(sample['annot'])
    
            # Pad the pseudo label
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

            ## For analysis
            # Compute iou between pseudo label and original label
            all_iou_pseu = []
            tp, fp, fn = 0, 0, 0
            for i, (annot, pseudo_annot, is_valid) in enumerate(zip(strong_u_sample['gt_annot'].numpy(), transformed_annots_padded, valid_mask)):
                annot = annot[annot[:, -1] != -1] # (ctr_z, ctr_y, ctr_x, d, h, w, space_z, space_y, space_x)
                if not is_valid:
                    fn += len(annot)
                    continue
                
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

                # Cheating, set FP to 0
                # for j in np.where(iou_pseu < 1e-3)[0]:
                #     transformed_annots_padded[i, j, ...] = -1
                
            if len(all_iou_pseu) > 0:
                avg_iou_pseu.update(np.mean(all_iou_pseu))
            avg_tp_pseu.update(tp)
            avg_fp_pseu.update(fp)
            avg_fn_pseu.update(fn)
            # transformed_annots_padded = transformed_annots_padded[valid_mask]
            # strong_u_sample['image'] = strong_u_sample['image'][valid_mask]
            strong_u_sample['annot'] = torch.from_numpy(transformed_annots_padded)
            
            background_mask = background_mask.view(bs, -1) # shape: (bs, num_points)
            # background_mask = background_mask[valid_mask]
            
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_pseu, cls_pos_pseu_loss, cls_neg_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss, outputs_pseu = unsupervised_train_one_step(args, model_s, strong_u_sample, background_mask, device)
            else:
                loss_pseu, cls_pos_pseu_loss, cls_neg_pseu_loss, shape_pseu_loss, offset_pseu_loss, iou_pseu_loss, outputs_pseu = unsupervised_train_one_step(args, model_s, strong_u_sample, background_mask, device)
            
            avg_pseu_cls_loss.update(cls_pos_pseu_loss.item() + cls_neg_pseu_loss.item())
            avg_pseu_cls_pos_loss.update(cls_pos_pseu_loss.item())
            avg_pseu_cls_neg_loss.update(cls_neg_pseu_loss.item())
            avg_pseu_shape_loss.update(shape_pseu_loss.item())
            avg_pseu_offset_loss.update(offset_pseu_loss.item())
            avg_pseu_iou_loss.update(iou_pseu_loss.item())
            avg_pseu_loss.update(loss_pseu.item())
            num_pseudo_nodules += len(strong_u_sample['annot'][strong_u_sample['annot'][..., -1] != -1])
            del outputs_pseu
            ### Labeled data
            try:
                labeled_sample = next(iter_l)
            except StopIteration:
                iter_l = iter(dataloader_l)
                labeled_sample = next(iter_l)
            
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model_s, labeled_sample, device)
            else:
                loss, cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss, outputs = train_one_step(args, model_s, labeled_sample, device)
            
            avg_cls_loss.update(cls_pos_loss.item() + cls_neg_loss.item())
            avg_cls_pos_loss.update(cls_pos_loss.item())
            avg_cls_neg_loss.update(cls_neg_loss.item())
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
            progress_bar.set_postfix(cls_pos_l = avg_cls_pos_loss.avg,
                                    cls_neg_l = avg_cls_neg_loss.avg,
                                    cls_pos_u = avg_pseu_cls_pos_loss.avg,
                                    cls_neg_u = avg_pseu_cls_neg_loss.avg,
                                    avg_iou_pseu = avg_iou_pseu.avg,
                                    num_u = num_pseudo_nodules,
                                    tp = avg_tp_pseu.sum,
                                    fp = avg_fp_pseu.sum,
                                    fn = avg_fn_pseu.sum)
            progress_bar.update()
            
            with torch.no_grad():
                # Update teacher model by exponential moving average
                for param, teacher_param in zip(model_s.parameters(), model_t.parameters()):
                    if param.requires_grad:
                        teacher_param.data.mul_(args.semi_ema_alpha).add_(param.data, alpha = 1 - args.semi_ema_alpha)
                # for (name_s, buffer_s), (name_t, buffer_t) in zip(model_s.named_buffers(), model_t.named_buffers()):
                #     if 'num_batches_tracked' in name_s:
                #         continue
                #     buffer_t.data.mul_(args.semi_ema_alpha).add_(buffer_s.data, alpha = 1 - args.semi_ema_alpha)
                    
            torch.cuda.empty_cache()
            
    recall = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fn_pseu.sum, 1e-3)
    precision = avg_tp_pseu.sum / max(avg_tp_pseu.sum + avg_fp_pseu.sum, 1e-3)
    f1 = 2 * recall * precision / max(recall + precision, 1e-3)
    metrics = {'loss': avg_loss.avg,
                'cls_loss': avg_cls_loss.avg,
                'cls_pos_loss': avg_cls_pos_loss.avg,
                'cls_neg_loss': avg_cls_neg_loss.avg,
                'shape_loss': avg_shape_loss.avg,
                'offset_loss': avg_offset_loss.avg,
                'iou_loss': avg_iou_loss.avg,
                'loss_pseu': avg_pseu_loss.avg,
                'cls_loss_pseu': avg_pseu_cls_loss.avg,
                'cls_pos_loss_pseu': avg_pseu_cls_pos_loss.avg,
                'cls_neg_loss_pseu': avg_pseu_cls_neg_loss.avg,
                'shape_loss_pseu': avg_pseu_shape_loss.avg,
                'offset_loss_pseu': avg_pseu_offset_loss.avg,
                'avg_iou_pseu': avg_iou_pseu.avg,
                'iou_loss_pseu': avg_pseu_iou_loss.avg,
                'num_pseudo_nodules':  num_pseudo_nodules,
                'pseu_recall': recall,
                'pseu_precision': precision,
                'pseudo_f1': f1,
                'pseu_tp': avg_tp_pseu.sum,
                'pseu_fp': avg_fp_pseu.sum,
                'pseu_fn': avg_fn_pseu.sum}
    return metrics