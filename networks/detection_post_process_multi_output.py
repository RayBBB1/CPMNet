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

    def get_scores(self, output, is_logits=True, lobe_mask=None):
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        
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

    def forward(self, output, device, is_logits=True, lobe_mask1=None, lobe_mask2=None):
        out1, out2 = output
        
        batch_size = out1['Cls'].size()[0]
        num_anchors = out1['Cls'].size()[2] * out1['Cls'].size()[3] * out1['Cls'].size()[4]
        
        topk_scores, topk_indices, pred_bboxes = self.get_scores(out1, is_logits, lobe_mask1)
        topk_scores2, topk_indices2, pred_bboxes2 = self.get_scores(out2, is_logits, lobe_mask2)
        
        topk_indices2 = topk_indices2 + num_anchors
        # Concatenate the topk scores and indices
        topk_scores = torch.cat((topk_scores, topk_scores2), dim=1) # (b, topk * 2)
        topk_indices = torch.cat((topk_indices, topk_indices2), dim=1) # (b, topk * 2)
        pred_bboxes = torch.cat((pred_bboxes, pred_bboxes2), dim=1) # (b, num_anchors * 2, 6)
        
        dets = (-torch.ones((batch_size, self.topk * 2, 8))).to(device)
        for j in range(batch_size):
            # Get indices of scores greater than threshold
            topk_score = topk_scores[j]
            topk_idx = topk_indices[j]
            keep_box_mask = (topk_score > self.threshold)
            keep_box_n = keep_box_mask.sum()
            
            if keep_box_n > 0:
                keep_topk_score = topk_score[keep_box_mask]
                keep_topk_idx = topk_idx[keep_box_mask]
                
                # 1, prob, ctr_z, ctr_y, ctr_x, d, h, w
                det = (-torch.ones((keep_box_n, 8))).to(device)
                det[:, 0] = 1
                det[:, 1] = keep_topk_score
                det[:, 2:] = pred_bboxes[j][keep_topk_idx]
            
                keep = nms_3D(det[:, 1:], overlap=self.nms_threshold, top_k=self.nms_topk)
                dets[j][:len(keep)] = det[keep.long()]

        if self.min_size > 0:
            dets_volumes = dets[:, :, 4] * dets[:, :, 5] * dets[:, :, 6]
            dets[dets_volumes < self.min_size] = -1
                
        return dets