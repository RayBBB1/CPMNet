from typing import Tuple, List, Union, Dict
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.box_utils import bbox_decode, make_anchors, zyxdhw2zyxzyx, project



class DetectionLoss(nn.Module):
    def __init__(self, 
                 crop_size=[96, 96, 96],
                 pos_target_topk = 7, 
                 pos_ignore_ratio = 3,
                 cls_num_neg = 10000,
                 cls_num_hard = 100,
                 cls_fn_weight = 4.0,
                 cls_fn_threshold = 0.8,
                 cls_neg_pos_ratio = 100,
                 cls_hard_fp_thrs1 = 0.5,
                 cls_hard_fp_thrs2 = 0.7,
                 cls_hard_fp_w1 = 1.5,
                 cls_hard_fp_w2 = 2.0,
                 cls_focal_alpha = 0.75,
                 cls_focal_gamma = 2.0,
                 max_reg=35):
        super(DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        
        self.cls_num_neg = cls_num_neg
        self.cls_num_hard = cls_num_hard
        self.cls_fn_weight = cls_fn_weight
        self.cls_fn_threshold = cls_fn_threshold
        
        self.cls_neg_pos_ratio = cls_neg_pos_ratio
        
        self.cls_hard_fp_thrs1 = cls_hard_fp_thrs1
        self.cls_hard_fp_thrs2 = cls_hard_fp_thrs2
        self.cls_hard_fp_w1 = cls_hard_fp_w1
        self.cls_hard_fp_w2 = cls_hard_fp_w2
        
        self.cls_focal_alpha = cls_focal_alpha
        self.cls_focal_gamma = cls_focal_gamma
        self.max_reg = max_reg
        
    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, neg_pos_ratio = 100, fn_weight = 4.0, fn_threshold = 0.8, 
                 hard_fp_thrs1 = 0.5, hard_fp_thrs2 = 0.7, hard_fp_w1 = 1.5, hard_fp_w2 = 2.0):
        """
        Calculates the classification loss using focal loss and binary cross entropy.

        Args:
            pred (torch.Tensor): The predicted logits of shape (b, num_points, 1)
            target: The target labels of shape (b, num_points, 1)
            mask_ignore: The mask indicating which pixels to ignore of shape (b, num_points, 1)
            alpha (float): The alpha factor for focal loss (default: 0.75)
            gamma (float): The gamma factor for focal loss (default: 2.0)
            num_neg (int): The maximum number of negative pixels to consider (default: 10000, if -1, use all negative pixels)
            num_hard (int): The number of hard negative pixels to keep (default: 100)
            ratio (int): The ratio of negative to positive pixels to consider (default: 100)
            fn_weight (float): The weight for false negative pixels (default: 4.0)
            fn_threshold (float): The threshold for considering a pixel as a false negative (default: 0.8)
            hard_fp_weight (float): The weight for hard false positive pixels (default: 2.0)
            hard_fp_threshold (float): The threshold for considering a pixel as a hard false positive (default: 0.7)
        Returns:
            torch.Tensor: The calculated classification loss
        """
        cls_pos_losses = []
        cls_neg_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            
            # Calculate the focal weight
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # Calculate the binary cross entropy loss
            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            if num_positive_pixels > 0:
                # Weight the hard false negatives(FN)
                FN_index = torch.lt(cls_prob, fn_threshold) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] *= fn_weight
                
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                    
                Positive_loss = cls_loss[record_targets == 1]
                Negative_loss = cls_loss[record_targets == 0]
                # Randomly sample negative pixels
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                    Negative_loss = Negative_loss[neg_idcs]
                    
                # Get the top k negative pixels
                _, keep_idx = torch.topk(Negative_loss, min(neg_pos_ratio * num_positive_pixels, len(Negative_loss))) 
                Negative_loss = Negative_loss[keep_idx] 
                
                # Calculate the loss
                num_positive_pixels = torch.clamp(num_positive_pixels.float(), min=1.0)
                Positive_loss = Positive_loss.sum() / num_positive_pixels
                Negative_loss = Negative_loss.sum() / num_positive_pixels
                cls_pos_losses.append(Positive_loss)
                cls_neg_losses.append(Negative_loss)
            else: # no positive pixels
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                
                # Randomly sample negative pixels
                Negative_loss = cls_loss[record_targets == 0]
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                    Negative_loss = Negative_loss[neg_idcs]
                
                # Get the top k negative pixels   
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                
                # Calculate the loss
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_neg_losses.append(Negative_loss)
                
        if len(cls_pos_losses) == 0:
            cls_pos_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_pos_loss = torch.sum(torch.stack(cls_pos_losses)) / batch_size
            
        if len(cls_neg_losses) == 0:
            cls_neg_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_neg_loss = torch.sum(torch.stack(cls_neg_losses)) / batch_size
        return cls_pos_loss, cls_neg_loss

    @staticmethod
    def quality_focal_loss(
          pred: torch.Tensor,
          target,
          score,
          mask_ignore,
          num_neg = 10000,
          num_hard = 100,
          neg_pos_ratio = 100,
          fn_weight = 4,
          fn_threshold = 0.5,
          hard_fp_thrs1 = -1,
          hard_fp_thrs2 = -1,
          hard_fp_w1 = -1,
          hard_fp_w2 = -1,
          beta=2.0,
          alpha = 0.75,
          gamma = 2.0):
        
        """
        Calculates the classification loss using focal loss and binary cross entropy.

        Args:
            pred (torch.Tensor): The predicted logits of shape (b, num_points, 1)
            target: The target labels of shape (b, num_points, 1)
            mask_ignore: The mask indicating which pixels to ignore of shape (b, num_points, 1)
            alpha (float): The alpha factor for focal loss (default: 0.75)
            gamma (float): The gamma factor for focal loss (default: 2.0)
            num_neg (int): The maximum number of negative pixels to consider (default: 10000, if -1, use all negative pixels)
            num_hard (int): The number of hard negative pixels to keep (default: 100)
            ratio (int): The ratio of negative to positive pixels to consider (default: 100)
            fn_weight (float): The weight for false negative pixels (default: 4.0)
            fn_threshold (float): The threshold for considering a pixel as a false negative (default: 0.8)
            hard_fp_weight (float): The weight for hard false positive pixels (default: 2.0)
            hard_fp_threshold (float): The threshold for considering a pixel as a hard false positive (default: 0.7)
        Returns:
            torch.Tensor: The calculated classification loss
        """
        cls_pos_losses = []
        cls_neg_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            score_b = score[j]
            mask_ignore_b = mask_ignore[j]
            # Calculate the focal weight
            # cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = pred_b.detach()
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)

            focal_weight = torch.where(torch.eq(target_b, 1.), score_b - cls_prob, cls_prob)
            # focal_weight = torch.where(torch.eq(target_b, 1.), score_b, alpha*torch.pow(pred_b, gamma))
            focal_weight = torch.pow(focal_weight, beta)
    
            # Calculate the binary cross entropy loss
            bce = F.binary_cross_entropy_with_logits(pred_b, score_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            if num_positive_pixels > 0:
                # Weight the hard false negatives(FN)
                FN_index = torch.lt(cls_prob, fn_threshold) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] *= fn_weight
                
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                    
                Positive_loss = cls_loss[record_targets == 1]
                Negative_loss = cls_loss[record_targets == 0]
                # Randomly sample negative pixels
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                    Negative_loss = Negative_loss[neg_idcs]
                    
                # Get the top k negative pixels
                _, keep_idx = torch.topk(Negative_loss, min(neg_pos_ratio * num_positive_pixels, len(Negative_loss))) 
                Negative_loss = Negative_loss[keep_idx] 
                
                # Calculate the loss
                num_positive_pixels = torch.clamp(num_positive_pixels.float(), min=1.0)
                Positive_loss = Positive_loss.sum() / num_positive_pixels
                Negative_loss = Negative_loss.sum() / num_positive_pixels
                cls_pos_losses.append(Positive_loss)
                cls_neg_losses.append(Negative_loss)
            else: # no positive pixels
                # Weight the hard false positives(FP)
                if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
                    hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
                    hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
                    cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                
                # Randomly sample negative pixels
                Negative_loss = cls_loss[record_targets == 0]
                if num_neg != -1:
                    neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                    Negative_loss = Negative_loss[neg_idcs]
                
                # Get the top k negative pixels   
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                
                # Calculate the loss
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_neg_losses.append(Negative_loss)
                
        if len(cls_pos_losses) == 0:
            cls_pos_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_pos_loss = torch.sum(torch.stack(cls_pos_losses)) / batch_size
            
        if len(cls_neg_losses) == 0:
            cls_neg_loss = torch.tensor(0.0, device=pred.device)
        else:
            cls_neg_loss = torch.sum(torch.stack(cls_neg_losses)) / batch_size
        return cls_pos_loss, cls_neg_loss

    @staticmethod
    def distribution_focal_loss(pred_shapes, target_shapes, predict_score, mask_ignore, target, max_reg=35):
        mask = target.clone()
        mask[mask_ignore.long()] = 0
        mask = mask.squeeze(-1).bool()
        if torch.sum(mask) > 0:
            cls_prob = torch.sigmoid(predict_score.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            pred_pos_cls_prob = cls_prob[mask].reshape(-1, 1).expand(-1, 3).reshape(-1)
            # print('pred, {}'.format(pred_shapes[mask].shape))
            # print('target, {}'.format(target_shapes[mask].shape))
            pred_shapes = pred_shapes[mask].reshape(-1, max_reg+1)
            # pred_shapes = torch.clamp(pred_shapes, 1e-4, 1.0 - 1e-4)
            target_shapes = target_shapes[mask].reshape(-1)
            disl = target_shapes.long()
            disr = disl + 1

            wl = disr.float() - target_shapes
            wr = target_shapes - disl.float()

            loss = F.cross_entropy(pred_shapes, disl, reduction='none') * wl \
            + F.cross_entropy(pred_shapes, disr, reduction='none') * wr
            # print('target', torch.sum(target))
            # print('mask', torch.sum(mask))
            # print(loss.shape)
            # loss = pred_pos_cls_prob*loss
            return torch.mean(loss)
        else:
            return torch.tensor(0.0, device=pred_shapes.device)

    @staticmethod
    def target_proprocess(annotations: torch.Tensor, 
                          device, 
                          input_size: List[int],
                          mask_ignore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the annotations to generate the targets for the network.
        In this function, we remove some annotations that the area of nodule is too small in the crop box. (Probably cropped by the edge of the image)
        
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
            input_size: List[int]
                A list of length 3 containing the (z, y, x) dimensions of the input.
            mask_ignore: torch.Tensor
                A zero tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        Returns: 
            A tuple of two tensors:
                (1) annotations_new: torch.Tensor
                    A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                    (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
                (2) mask_ignore: torch.Tensor
                    A tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        """
        batch_size = annotations.shape[0]
        annotations_new = -1 * torch.ones_like(annotations, device=device)
        for sample_i in range(batch_size):
            annots = annotations[sample_i]
            gt_bboxes = annots[annots[:, -1] > -1] # -1 means ignore, it is used to make each sample has same number of bbox (pad with -1)
            bbox_annotation_target = []
            
            crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]], device=device)
            for s in range(len(gt_bboxes)):
                each_label = gt_bboxes[s] # (z_ctr, y_ctr, x_ctr, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1)
                # coordinate convert zmin, ymin, xmin, d, h, w
                z1 = (torch.max(each_label[0] - each_label[3]/2., crop_box[0]))
                y1 = (torch.max(each_label[1] - each_label[4]/2., crop_box[1]))
                x1 = (torch.max(each_label[2] - each_label[5]/2., crop_box[2]))

                z2 = (torch.min(each_label[0] + each_label[3]/2., crop_box[3]))
                y2 = (torch.min(each_label[1] + each_label[4]/2., crop_box[4]))
                x2 = (torch.min(each_label[2] + each_label[5]/2., crop_box[5]))
                
                nd = torch.clamp(z2 - z1, min=0.0)
                nh = torch.clamp(y2 - y1, min=0.0)
                nw = torch.clamp(x2 - x1, min=0.0)
                if nd * nh * nw == 0:
                    continue
                percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
                if (percent > 0.1) and (nw*nh*nd >= 15):
                    spacing_z, spacing_y, spacing_x = each_label[6:9]
                    bbox = torch.from_numpy(np.array([float(z1 + 0.5 * nd), float(y1 + 0.5 * nh), float(x1 + 0.5 * nw), float(nd), float(nh), float(nw), float(spacing_z), float(spacing_y), float(spacing_x), 0])).to(device)
                    bbox_annotation_target.append(bbox.view(1, 10))
                else:
                    mask_ignore[sample_i, 0, int(z1) : int(torch.ceil(z2)), int(y1) : int(torch.ceil(y2)), int(x1) : int(torch.ceil(x2))] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                annotations_new[sample_i, :len(bbox_annotation_target)] = bbox_annotation_target
        return annotations_new, mask_ignore
    
    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7, is_norm=False, is_enhance=False):
        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
                (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

        # Union Area
        union = w1 * h1 * d1 + w2 * h2 * d2 - inter

        # IoU
        iou = inter / union
        if DIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
            cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
            c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
            + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
            if is_norm:
                return ((iou - rho2 / c2) + 1)/2  # DIoU
            else:
                return iou - rho2 / c2  # DIoU
        if is_enhance:
            # print('before',iou)
            t = torch.tensor(0.3, device=iou.device)
            offset = torch.zeros_like(iou, device=iou.device)
            offset = torch.where(torch.gt(offset, 0.5), 0, 0.5)
            iou += offset
            iou = torch.pow(iou, 1/t)/(torch.pow(iou, 1/t)+torch.pow(1-iou, 1/t))
            iou -= offset
            print('after',iou)
        return iou  # IoU
    
    @staticmethod
    def distance_score(box1, box2, eps = 1e-7):
        box1 = zyxdhw2zyxzyx(box1)
        box2 = zyxdhw2zyxzyx(box2)
        # Get the coordinates of bounding boxes
        b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
        b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)

        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
        c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
        + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
        return 1 - (rho2/c2)
    
    @staticmethod
    def get_pos_target(annotations: torch.Tensor,
                       anchor_points: torch.Tensor,
                       stride: torch.Tensor,
                       pos_target_topk = 7, 
                       ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
        """Get the positive targets for the network.
        Steps:
            1. Calculate the distance between each annotation and each anchor point.
            2. Find the top k anchor points with the smallest distance for each annotation.
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format: (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1).
            anchor_points: torch.Tensor
                A tensor of shape (num_of_point, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride: torch.Tensor
                A tensor of shape (1,1,3) containing the strides of each dimension in format (z, y, x).
        """
        batchsize, num_of_annots, _ = annotations.size()
        # -1 means ignore, larger than 0 means positive
        # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
        # indicating whether the annotation is ignored or not.
        mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
        # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
        ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        
        sp = annotations[:, :, 6:9] # spacing, shape = (b, num_annotations, 3)
        sp = sp.unsqueeze(-2) # shape = (b, num_annotations, 1, 3)
        
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1)) # (b, num_annotation, num_of_points)
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
        # mask_topk = F.one_hot(topk_inds[:, :, :pos_target_topk], distance.size()[-1]).sum(-2) # (b, num_annotation, num_of_points), the value is 1 or 0
        # mask_ignore = -1 * F.one_hot(topk_inds[:, :, pos_target_topk:], distance.size()[-1]).sum(-2) # the value is -1 or 0
        mask_topk = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, :pos_target_topk], 1) # (b, num_annotation, num_of_points)
        mask_ignore = torch.zeros_like(distance, device=distance.device).scatter_(-1, topk_inds[:, :, pos_target_topk:], -1) # (b, num_annotation, num_of_points)
        
        # the value is 1 or 0, shape= (b, num_annotations, num_of_points)
        # mask_gt is 1 mean the annotation is not ignored, 0 means the annotation is ignored
        # mask_topk is 1 means the point is assigned to positive
        # mask_topk * mask_gt.unsqueeze(-1) is 1 means the point is assigned to positive and the annotation is not ignored
        mask_pos = mask_topk * mask_gt.unsqueeze(-1) 
        
        mask_ignore = mask_ignore * mask_gt.unsqueeze(-1) # the value is -1 or 0, shape= (b, num_annotations, num_of_points)
        gt_idx = mask_pos.argmax(-2) # shape = (b, num_of_points), it indicates each point matches which annotation
        
        # Flatten the batch dimension
        batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None] # (b, 1)
        gt_idx = gt_idx + batch_ind * num_of_annots
        
        # Generate the targets of each points
        target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
        target_offset = target_ctr - anchor_points
        target_shape = shape.view(-1, 3)[gt_idx]
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    
    def compute_score(self,
                    pred_bboxes,
                    target_bboxes,
                    target_scores):
        """
        Args:
            pred_bboxes: torch.Tensor, shape:(b, num_anchor, 6)
            target_bboxes: torch.Tensor, shape:(b, num_anchor, 6)
            target_scores: torch.Tensor, shape:(b, num_anchor, 1)
        Return:
            scores: torch.Tensor, shape:(b, num_anchor, 1), the value is iou score beteween pred_bbox and target_bbox
        """
        fg_mask = target_scores.squeeze(-1).bool()
        scores = torch.zeros_like(target_scores)
        scores[fg_mask] = self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], DIoU=False, is_enhance=False)
        # scores[fg_mask] = self.distance_score(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        return scores

    def forward(self, 
                output: Dict[str, torch.Tensor], 
                annotations: torch.Tensor,
                device):
        """
        Args:
            output: Dict[str, torch.Tensor], the output of the model.
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
        """
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1) # (b, 1, num_points)
        pred_shapes_distribution = Shape.view(batch_size, 3*(self.max_reg+1), -1) # (b, 3*(self.max_reg+1), num_points)
        # pred_offsets = Offset.view(batch_size, 3, -1)
        pred_offests_distribution = Offset.view(batch_size, 3*(self.max_reg+1), -1) # (b, 3*(self.max_reg+1), num_points)
        # (b, num_points, 1 or 3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes_distribution = pred_shapes_distribution.permute(0, 2, 1).contiguous()
        pred_shapes = project(pred_shapes_distribution)
        pred_offests_distribution = pred_offests_distribution.permute(0, 2, 1).contiguous()
        pred_offsets = project(pred_offests_distribution)
        
        # process annotations
        process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, target_mask_ignore)
        target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()

        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # shape = (num_anchors, 3)

        # assigned points and targets (target bboxes zyxdhw)
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
                                                                                                     anchor_points = anchor_points,
                                                                                                     stride = stride_tensor[0].view(1, 1, 3), 
                                                                                                     pos_target_topk = self.pos_target_topk,
                                                                                                     ignore_ratio = self.pos_ignore_ratio)
        
        # predict bboxes (zyxdhw)
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor) # shape = (b, num_anchors, 6) #
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        mask_ignore = mask_ignore.int()
        
        scores = self.compute_score(pred_bboxes=pred_bboxes, target_bboxes=target_bboxes, target_scores=target_scores)
        
        # cls_pos_loss, cls_neg_loss = self.cls_loss(pred = pred_scores, 
        #                                            target = target_scores, 
        #                                            mask_ignore = mask_ignore, 
        #                                            neg_pos_ratio = self.cls_neg_pos_ratio,
        #                                            num_hard = self.cls_num_hard, 
        #                                            num_neg = self.cls_num_neg,
        #                                            fn_weight = self.cls_fn_weight, 
        #                                            fn_threshold = self.cls_fn_threshold,
        #                                            hard_fp_thrs1=self.cls_hard_fp_thrs1,
        #                                            hard_fp_thrs2=self.cls_hard_fp_thrs2,
        #                                             hard_fp_w1=self.cls_hard_fp_w1,
        #                                             hard_fp_w2=self.cls_hard_fp_w2,
        #                                             alpha=self.cls_focal_alpha,
        #                                             gamma=self.cls_focal_gamma)

        cls_pos_loss, cls_neg_loss = self.quality_focal_loss(pred = pred_scores, 
                                                   target = target_scores, 
                                                   score= scores,
                                                   mask_ignore = mask_ignore, 
                                                   num_neg = self.cls_num_neg,
                                                   num_hard = self.cls_num_hard, 
                                                   neg_pos_ratio = self.cls_neg_pos_ratio,
                                                   fn_weight = self.cls_fn_weight, 
                                                   fn_threshold = 0.5,
                                                   hard_fp_thrs1=-1,
                                                   hard_fp_thrs2=-1,
                                                    hard_fp_w1=-1,
                                                    hard_fp_w2=-1,
                                                    beta=self.cls_focal_gamma)
        reg_loss = self.distribution_focal_loss(pred_shapes=pred_shapes_distribution, predict_score=pred_scores,  target_shapes=target_shape, mask_ignore=mask_ignore, target=target_scores)
        offset_loss = self.distribution_focal_loss(pred_shapes=pred_offests_distribution, predict_score=pred_scores, target_shapes=target_offset, mask_ignore=mask_ignore, target=target_scores)
        # Only calculate the loss of positive samples                                 
        fg_mask = target_scores.squeeze(-1).bool()
        if fg_mask.sum() == 0:
            # reg_loss = torch.tensor(0.0, device=device)
            # offset_loss = torch.tensor(0.0, device=device)
            iou_loss = torch.tensor(0.0, device=device)
        else:
            # print('fg_mask, {} {}'.format(fg_mask.shape, torch.sum(fg_mask)))
            # print('pred_shapes[fg_mask], {}'.format(pred_shapes[fg_mask].shape))
            # print('target_shape[fg_mask], {}'.format(target_shape[fg_mask].shape))
            # reg_loss = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            # offset_loss = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_loss = 1 - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss