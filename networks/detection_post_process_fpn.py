import torch
import torch.nn as nn
from utils.box_utils import nms_3D, bbox_decode, make_anchors
from typing import List

class DetectionPostprocess(nn.Module):
    def __init__(self, topk: int=60, threshold: float=0.15, nms_threshold: float=0.05, nms_topk: int=20, crop_size: List[int]=[96, 96, 96], min_size: int=-1):
        super(DetectionPostprocess, self).__init__()
        self.topk = topk
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = nms_topk
        self.crop_size = crop_size
        self.min_size = min_size

    def get_scores(self, Cls, Shape, Offset, stride, is_logits=True, lobe_mask=None):
        # Cls = output['Cls']
        # Shape = output['Shape']
        # Offset = output['Offset']
        
        if lobe_mask is not None:
            if is_logits:
                Cls[lobe_mask == 0] = -20 # ignore the lobe 0, -20 indicates the background, sigmoid(-20) is close to 0
            else:
                Cls[lobe_mask == 0] = 1e-4
        
        batch_size = Cls.size()[0]
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous().squeeze(dim=2)
        if is_logits:
            pred_scores = pred_scores.sigmoid()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
        # recale to input_size
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor)
        # Get the topk scores and indices
        topk_scores, topk_indices = torch.topk(pred_scores, self.topk, dim=-1, largest=True)

        return topk_scores, topk_indices, pred_bboxes

    def forward(self, output, device, is_logits=True, lobe_mask=None):
        # out1, out2 = output
        strides = [4, 8, 16]
        Clses = output['Cls']
        Shapes = output['Shape']
        Offsets = output['Offset']
        batch_size = Clses[0].size()[0]
        num_outputs = len(Clses)
        shift_bias = 0
        tot_topk_scores, tot_topk_indices, tot_pred_bboxes = None, None, None
        for output_index in range(num_outputs):
            num_anchors = Clses[output_index].size()[2] * Clses[output_index].size()[3] * Clses[output_index].size()[4]
            topk_scores, topk_indices, pred_bboxes = self.get_scores(Clses[output_index], Shapes[output_index], Offsets[output_index], strides[output_index], is_logits, lobe_mask)
            # topk_scores2, topk_indices2, pred_bboxes2 = self.get_scores(out2, is_logits, lobe_mask2)
            topk_indices = topk_indices + shift_bias
            # print('shift_bias', shift_bias)
            shift_bias += num_anchors
            if output_index > 0:
                # Concatenate the topk scores and indices
                tot_topk_scores = torch.cat((tot_topk_scores, topk_scores), dim=1) # (b, topk * 2)
                tot_topk_indices = torch.cat((tot_topk_indices, topk_indices), dim=1) # (b, topk * 2)
                tot_pred_bboxes = torch.cat((tot_pred_bboxes, pred_bboxes), dim=1) # (b, num_anchors * 2, 6)
            else:
                tot_topk_scores = topk_scores
                tot_topk_indices = topk_indices
                tot_pred_bboxes = pred_bboxes
        # print('tot_topk_scores', tot_topk_scores.shape)
        # print('tot_topk_indices', tot_topk_indices.shape)
        # print('tot_pred_bboxes', tot_pred_bboxes.shape)
        dets = (-torch.ones((batch_size, self.topk * 3, 8))).to(device)
        for j in range(batch_size):
            # Get indices of scores greater than threshold
            topk_score = tot_topk_scores[j]
            topk_idx = tot_topk_indices[j]
            keep_box_mask = (topk_score > self.threshold)
            keep_box_n = keep_box_mask.sum()
            
            if keep_box_n > 0:
                keep_topk_score = topk_score[keep_box_mask]
                keep_topk_idx = topk_idx[keep_box_mask]
                
                # 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                det = (-torch.ones((keep_box_n, 8))).to(device)
                det[:, 0] = 1
                det[:, 1] = keep_topk_score
                det[:, 2:] = tot_pred_bboxes[j][keep_topk_idx]
            
                keep = nms_3D(det[:, 1:], overlap=self.nms_threshold, top_k=self.nms_topk)
                dets[j][:len(keep)] = det[keep.long()]

        if self.min_size > 0:
            dets_volumes = dets[:, :, 4] * dets[:, :, 5] * dets[:, :, 6]
            dets[dets_volumes < self.min_size] = -1
                
        return dets