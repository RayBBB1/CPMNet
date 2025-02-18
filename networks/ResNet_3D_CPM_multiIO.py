from typing import Tuple, List, Union, Dict
import random
import math
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.box_utils import nms_3D
from .modules import SELayer, Identity, ConvBlock, act_layer, norm_layer3d

LOSS_WEIGHTS = [3/4, 1/4]

def bbox_decode(anchor_points: torch.Tensor, pred_offsets: torch.Tensor, pred_shapes: torch.Tensor, stride_tensor: torch.Tensor, dim=-1) -> torch.Tensor:
    """Apply the predicted offsets and shapes to the anchor points to get the predicted bounding boxes.
    anchor_points is the center of the anchor boxes, after applying the stride, new_center = (center + pred_offsets) * stride_tensor
    Args:
        anchor_points: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
        pred_offsets: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the predicted offsets in the format (dz, dy, dx).
        pred_shapes: torch.Tensor
            A tensor of shape (bs, num_anchors, 3) containing the predicted shapes in the format (d, h, w).
        stride_tensor: torch.Tensor
            A tensor of shape (bs, 3) containing the strides of each dimension in format (z, y, x).
    Returns:
        A tensor of shape (bs, num_anchors, 6) containing the predicted bounding boxes in the format (z, y, x, d, h, w).
    """
    center_zyx = (anchor_points + pred_offsets) * stride_tensor
    return torch.cat((center_zyx, 2*pred_shapes), dim)  # zyxdhw bbox

class BasicBlockNew(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_type='batchnorm', act_type='ReLU', se=True):
        super(BasicBlockNew, self).__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               act_type=act_type, norm_type=norm_type)

        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=1,
                               act_type='none', norm_type=norm_type)

        if in_channels == out_channels and stride == 1:
            self.res = Identity()
        elif in_channels != out_channels and stride == 1:
            self.res = ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type)
        elif in_channels != out_channels and stride > 1:
            self.res = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                ConvBlock(in_channels, out_channels, kernel_size=1, act_type='none', norm_type=norm_type))

        if se:
            self.se = SELayer(out_channels)
        else:
            self.se = Identity()

        self.act = act_layer(act_type)

    def forward(self, x):
        ident = self.res(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)

        x += ident
        x = self.act(x)

        return x

class LayerBasic(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, stride=1, norm_type='batchnorm', act_type='ReLU', se=False):
        super(LayerBasic, self).__init__()
        self.n_stages = n_stages
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = in_channels
                stride = stride
            else:
                input_channel = out_channels
                stride = 1

            ops.append(
                BasicBlockNew(input_channel, out_channels, stride=stride, norm_type=norm_type, act_type=act_type, se=se))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='batchnorm', act_type='ReLU'):
        super(DownsamplingConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, padding=0, stride=stride, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = act_layer(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm_type='batchnorm', act_type='ReLU'):
        super(UpsamplingDeconvBlock, self).__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=stride, padding=0, stride=stride,
                                       bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = act_layer(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ClsRegHead(nn.Module):
    def __init__(self, in_channels, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='LeakyReLU'):
        super(ClsRegHead, self).__init__()

        conv_s = []
        conv_r = []
        conv_o = []
        for i in range(conv_num):
            if i == 0:
                conv_s.append(ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
                conv_r.append(ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
                conv_o.append(ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_s.append(ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
                conv_r.append(ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
                conv_o.append(ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        
        self.conv_s = nn.Sequential(*conv_s)
        self.conv_r = nn.Sequential(*conv_r)
        self.conv_o = nn.Sequential(*conv_o)
        
        self.cls_output = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)
        self.shape_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)
        self.offset_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        Shape = self.shape_output(self.conv_r(x))
        Offset = self.offset_output(self.conv_o(x))
        Cls = self.cls_output(self.conv_s(x))
        dict1 = {}
        dict1['Cls'] = Cls
        dict1['Shape'] = Shape
        dict1['Offset'] = Offset
        return dict1

class Resnet18(nn.Module):
    def __init__(self, n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,
                 norm_type='batchnorm', head_norm='batchnorm', act_type='ReLU', se=False, first_stride=(2, 2, 2), detection_loss=None, device=None):
        super(Resnet18, self).__init__()
        self.detection_loss = detection_loss
        self.device = device

        self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=first_stride, norm_type=norm_type, act_type=act_type)

        self.block1 = LayerBasic(n_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
        self.block1_dw = DownsamplingConvBlock(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = LayerBasic(n_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block2_dw = DownsamplingConvBlock(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = LayerBasic(n_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block3_dw = DownsamplingConvBlock(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        self.block4 = LayerBasic(n_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type, se=se)

        self.block33_up = UpsamplingDeconvBlock(n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_res = LayerBasic(1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block33 = LayerBasic(2, n_filters[2] * 2, n_filters[2], norm_type=norm_type, act_type=act_type, se=se)

        self.block22_up = UpsamplingDeconvBlock(n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_res = LayerBasic(1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block22 = LayerBasic(2, n_filters[1] * 2, n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        
        self.head16 = ClsRegHead(in_channels=n_filters[2], feature_size=n_filters[2], conv_num=3, norm_type=head_norm, act_type=act_type)
        self.head32 = ClsRegHead(in_channels=n_filters[1], feature_size=n_filters[1], conv_num=3, norm_type=head_norm, act_type=act_type)
        
        self._init_weight()

    def forward(self, inputs):
        if self.training:
            x, labels = inputs
        else:
            x = inputs
        "input encode"
        x = self.in_conv(x)
        x = self.in_dw(x)
        
        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x = self.block4(x)

        "decode"
        x = self.block33_up(x)
        x3 = self.block33_res(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)

        if self.training:
            out16 = self.head16(x)
        
        x = self.block22_up(x)
        x2 = self.block22_res(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)

        out32 = self.head32(x)
        if self.training:
            cls_loss16, shape_loss16, offset_loss16, iou_loss16 = self.detection_loss(out16, labels, device=self.device, scale=0.5)
            cls_loss32, shape_loss32, offset_loss32, iou_loss32 = self.detection_loss(out32, labels, device=self.device)
            
            cls_loss = cls_loss32 * LOSS_WEIGHTS[0] + cls_loss16 * LOSS_WEIGHTS[1]
            shape_loss = shape_loss32 * LOSS_WEIGHTS[0] + shape_loss16 * LOSS_WEIGHTS[1]
            offset_loss = offset_loss32 * LOSS_WEIGHTS[0] + offset_loss16 * LOSS_WEIGHTS[1]
            iou_loss = iou_loss32 * LOSS_WEIGHTS[0] + iou_loss16 * LOSS_WEIGHTS[1]
            
            return cls_loss, shape_loss, offset_loss, iou_loss
        return out32

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        prior = 0.01
        nn.init.constant_(self.head32.cls_output.weight, 0)
        nn.init.constant_(self.head32.cls_output.bias, -math.log((1.0 - prior) / prior))

        nn.init.constant_(self.head32.shape_output.weight, 0)
        nn.init.constant_(self.head32.shape_output.bias, 0.5)

        nn.init.constant_(self.head32.offset_output.weight, 0)
        nn.init.constant_(self.head32.offset_output.bias, 0.05)

def make_anchors(feat: torch.Tensor, input_size: List[float], grid_cell_offset=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from a feature.
    Returns:
        num_anchor is the number of anchors in the feature map, which is d * h * w.
        A tuple of two tensors:
            anchor_points: torch.Tensor
                A tensor of shape (num_anchors, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride_tensor: torch.Tensor
                A tensor of shape (num_anchors, 3) containing the strides of the anchor points, the strides of each anchor point are same.
    """
    dtype, device = feat.dtype, feat.device
    _, _, d, h, w = feat.shape
    strides = torch.tensor([input_size[0] / d, 
                            input_size[1] / h, 
                            input_size[2] / w]).type(dtype).to(device)
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z
    anchor_points = torch.cartesian_prod(sz, sy, sx)
    stride_tensor = strides.repeat(d * h * w, 1)
    return anchor_points, stride_tensor

class DetectionLoss(nn.Module):
    def __init__(self, 
                 crop_size=[64, 128, 128], 
                 pos_target_topk = 7, 
                 pos_ignore_ratio = 3,
                 spacing=[2.0, 1.0, 1.0]):
        super(DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        self.spacing = np.array(spacing)

    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, ratio = 100):
        """
        Args:
            pred: torch.Tensor
                The predicted logits of shape (b, num_points, 1)
        """
        classification_losses = []
        batch_size = pred.shape[0]
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
            mask_ignore_b = mask_ignore[j]
            
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
            record_targets = target_b.clone()
            if num_positive_pixels > 0:
                FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(cls_prob, 0.8) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] = FN_weights * cls_loss[FN_index == 1]
                
                Negative_loss = cls_loss[record_targets == 0]
                Positive_loss = cls_loss[record_targets == 1]
                
                neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
                Negative_loss = Negative_loss[neg_idcs] 
                _, keep_idx = torch.topk(Negative_loss, min(ratio * num_positive_pixels, len(Negative_loss))) 
                Negative_loss = Negative_loss[keep_idx] 
                
                Positive_loss = Positive_loss.sum()
                Negative_loss = Negative_loss.sum()
                cls_loss = Positive_loss + Negative_loss

            else:
                Negative_loss = cls_loss[record_targets == 0]
                neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
                Negative_loss = Negative_loss[neg_idcs]
                assert len(Negative_loss) > num_hard
                _, keep_idx = torch.topk(Negative_loss, num_hard)
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_loss = Negative_loss
            classification_losses.append(cls_loss / torch.clamp(num_positive_pixels.float(), min=1.0))
        return torch.mean(torch.stack(classification_losses))
    
    @staticmethod
    def target_proprocess(annotations: torch.Tensor, 
                          device, 
                          input_size: List[int],
                          mask_ignore: torch.Tensor,
                          scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the annotations to generate the targets for the network.
        In this function, we remove some annotations that the area of nodule is too small in the crop box. (Probably cropped by the edge of the image)
        
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 7) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
            input_size: List[int]
                A list of length 3 containing the (z, y, x) dimensions of the input.
            mask_ignore: torch.Tensor
                A zero tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        Returns: 
            A tuple of two tensors:
                (1) annotations_new: torch.Tensor
                    A tensor of shape (batch_size, num_annotations, 7) containing the annotations in the format:
                    (ctr_z, ctr_y, ctr_x, d, h, w, 0 or -1). The last index -1 means the annotation is ignored.
                (2) mask_ignore: torch.Tensor
                    A tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
        """
        batch_size = annotations.shape[0]
        annotations_new = -1 * torch.ones_like(annotations).to(device)
        for sample_i in range(batch_size):
            annots = annotations[sample_i]
            gt_bboxes = annots[annots[:, -1] > -1] # -1 means ignore, it is used to make each sample has same number of bbox (pad with -1)
            bbox_annotation_target = []
            
            crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]]).to(device) # (z_ctr, y_ctr, x_ctr, d, h, w)
            for s in range(len(gt_bboxes)):
                each_label = gt_bboxes[s] # (z_ctr, y_ctr, x_ctr, d, h, w)
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
                    bbox = torch.from_numpy(np.array([float(z1 + 0.5 * nd), float(y1 + 0.5 * nh), float(x1 + 0.5 * nw), float(nd), float(nh), float(nw), 0])).to(device)
                    bbox_annotation_target.append(bbox.view(1, 7))
                else:
                    z1 = z1 * scale
                    y1 = y1 * scale
                    x1 = x1 * scale
                    z2 = z2 * scale
                    y2 = y2 * scale
                    mask_ignore[sample_i, 0, int(z1) : int(torch.ceil(z2)), int(y1) : int(torch.ceil(y2)), int(x1) : int(torch.ceil(x2))] = -1
            if len(bbox_annotation_target) > 0:
                bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
                bbox_annotation_target = bbox_annotation_target * scale
                annotations_new[sample_i, :len(bbox_annotation_target)] = bbox_annotation_target
        return annotations_new, mask_ignore
    
    @staticmethod
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
        def zyxdhw2zyxzyx(box, dim=-1):
            ctr_zyx, dhw = torch.split(box, 3, dim)
            z1y1x1 = ctr_zyx - dhw/2
            z2y2x2 = ctr_zyx + dhw/2
            return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox
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
            return iou - rho2 / c2  # DIoU
        return iou  # IoU
    
    @staticmethod
    def get_pos_target(annotations: torch.Tensor,
                       anchor_points: torch.Tensor,
                       stride: torch.Tensor,
                       spacing: np.ndarray,
                       pos_target_topk = 7, 
                       ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
        """Get the positive targets for the network.
        Steps:
            1. Calculate the distance between each annotation and each anchor point.
            2. Find the top k anchor points with the smallest distance for each annotation.
        Args:
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 7) containing the annotations in the format: (ctr_z, ctr_y, ctr_x, d, h, w, 0 or -1)
            anchor_points: torch.Tensor
                A tensor of shape (num_of_point, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
            stride: torch.Tensor
                A tensor of shape (1,1,3) containing the strides of each dimension in format (z, y, x).
            spacing: np.Array
                A ndarray of shape (3,) containing the spacing of each dimension in format (z, y, x).
        """
        batchsize, num_of_annots, _ = annotations.size()
        # -1 means ignore, larger than 0 means positive
        # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
        # indicating whether the annotation is ignored or not.
        mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
        # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
        ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
        shape = annotations[:, :, 3:6] / 2 # half d h w
        
        sp = torch.from_numpy(spacing).to(ctr_gt_boxes.device).view(1, 1, 1, 3)
        
        # distance (b, n_max_object, anchors)
        distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1))
        _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
        mask_topk = F.one_hot(topk_inds[:, :, :pos_target_topk], distance.size()[-1]).sum(-2) # (b, num_annotation, num_of_points), the value is 1 or 0
        mask_ignore = -1 * F.one_hot(topk_inds[:, :, pos_target_topk:], distance.size()[-1]).sum(-2) # the value is -1 or 0
        
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
        
        target_bboxes = annotations[:, :, :-1].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
    def forward(self, 
                output: Dict[str, torch.Tensor], 
                annotations: torch.Tensor,
                device,
                scale=1.0):
        """
        Args:
            output: Dict[str, torch.Tensor], the output of the model.
            annotations: torch.Tensor
                A tensor of shape (batch_size, num_annotations, 7) containing the annotations in the format:
                (ctr_z, ctr_y, ctr_x, d, h, w, 0 or -1). The last index -1 means the annotation is ignored.
            device: torch.device, the device of the model.
        """
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        target_mask_ignore = torch.zeros(Cls.size()).to(device)
        
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1) # (b, 1, num_points)
        pred_shapes = Shape.view(batch_size, 3, -1) # (b, 3, num_points)
        pred_offsets = Offset.view(batch_size, 3, -1)
        # (b, num_points, 1 or 3)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
        # process annotations
        process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, target_mask_ignore)
        target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
        target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
        # generate center points. Only support single scale feature
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # shape = (num_anchors, 3)
        # predict bboxes (zyxdhw)
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor) # shape = (b, num_anchors, 6)
        # assigned points and targets (target bboxes zyxdhw)
        scaled_pos_target_topk = min(int(self.pos_target_topk * scale) + 1, self.pos_target_topk)
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
                                                                                                     anchor_points = anchor_points,
                                                                                                     stride = stride_tensor[0].view(1, 1, 3), 
                                                                                                     spacing = self.spacing, 
                                                                                                     pos_target_topk = scaled_pos_target_topk,
                                                                                                     ignore_ratio = self.pos_ignore_ratio)
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        fg_mask = target_scores.squeeze(-1).bool()
        classification_losses = self.cls_loss(pred_scores, target_scores, mask_ignore.int())
        if fg_mask.sum() == 0:
            reg_losses = torch.tensor(0).float().to(device)
            offset_losses = torch.tensor(0).float().to(device)
            iou_losses = torch.tensor(0).float().to(device)
        else:
            reg_losses = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_losses = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_losses = - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return classification_losses, reg_losses, offset_losses, iou_losses

class DetectionPostprocess(nn.Module):
    def __init__(self, topk: int=60, threshold: float=0.15, nms_threshold: float=0.05, nms_topk: int=20, crop_size: List[int]=[64, 96, 96]):
        super(DetectionPostprocess, self).__init__()
        self.topk = topk
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = nms_topk
        self.crop_size = crop_size

    def forward(self, output, device):
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
        pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
        pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
        # recale to input_size
        pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor)
        # Get the topk scores and indices
        topk_scores, topk_indices = torch.topk(pred_scores.squeeze(), self.topk, dim=-1, largest=True)
        
        dets = (-torch.ones((batch_size, self.topk, 8))).to(device)
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
        return dets