from typing import Tuple, List, Union, Dict
import random
import math
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.box_utils import nms_3D
from .modules import SELayer, Identity, ConvBlock, act_layer, norm_layer3d

def zyxdhw2zyxzyx(box, dim=-1):
    ctr_zyx, dhw = torch.split(box, 3, dim)
    z1y1x1 = ctr_zyx - dhw/2
    z2y2x2 = ctr_zyx + dhw/2
    return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox

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

class ELAN_Layer(nn.Module):
    def __init__(self, n_stages: int, in_channels: int, out_channels: int, stride=1, norm_type='batchnorm', act_type='ReLU', se=False):
        super(ELAN_Layer, self).__init__()
        self.n_stages = n_stages
        # feat_size = out_channels // 2
        feat_size = out_channels
        
        self.stem1 = ConvBlock(in_channels // 2, feat_size, kernel_size=1, norm_type=norm_type, act_type='none')
        self.stem2 = ConvBlock(in_channels // 2, feat_size, kernel_size=1, norm_type=norm_type, act_type='none')
        for i in range(n_stages):
            setattr(self, f'conv{i}', BasicBlockNew(feat_size, feat_size, norm_type=norm_type, act_type=act_type, se=se))
        
        self.trans = ConvBlock(feat_size * (n_stages + 2), out_channels, kernel_size=1, act_type=act_type, norm_type=norm_type)
        
    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        y = [self.stem1(x1), self.stem2(x2)]
        for i in range(self.n_stages):
            y.append(getattr(self, f'conv{i}')(y[-1]))
        return self.trans(torch.cat(y, 1))

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='none', act_type='ReLU'):
        super(StemBlock, self).__init__()
        feat_size = out_channels // 2
        self.conv1_1 = ConvBlock(in_channels, feat_size, kernel_size=1, norm_type=norm_type, act_type='none')
        
        self.conv2_1 = ConvBlock(in_channels, feat_size, kernel_size=1, norm_type=norm_type, act_type='none')
        self.conv2_2  = ConvBlock(feat_size, feat_size, kernel_size=3, norm_type=norm_type, act_type='none')
        
        self.conv3_1 = ConvBlock(in_channels, feat_size, kernel_size=1, norm_type=norm_type, act_type='none')
        self.conv3_2 = ConvBlock(feat_size, feat_size, kernel_size=5, norm_type=norm_type, act_type='none')
        
        self.trans = ConvBlock(feat_size * 3, out_channels, kernel_size=1, norm_type=norm_type, act_type=act_type)
    
    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv2_2(self.conv2_1(x))
        x3 = self.conv3_2(self.conv3_1(x))
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.trans(x)
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

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, pool_type='max',
                 norm_type='batchnorm', act_type='ReLU'):
        super(DownsamplingBlock, self).__init__()

        if pool_type == 'avg':
            self.down = nn.AvgPool3d(kernel_size=stride, stride=stride)
        else:
            self.down = nn.MaxPool3d(kernel_size=stride, stride=stride)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.down(x)
        if hasattr(self, 'conv'):
            x = self.conv(x)
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

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=2, mode='nearest', norm_type='batchnorm',
                 act_type='ReLU'):
        super(UpsamplingBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        if hasattr(self, 'conv'):
            x = self.conv(x)
        x = self.up(x)
        return x

class ASPP(nn.Module):
    def __init__(self, channels, ratio=2,
                 dilations=[1, 1, 2, 3],
                 norm_type='batchnorm', act_type='ReLU'):
        super(ASPP, self).__init__()
        # assert dilations[0] == 1, 'The first item in dilations should be `1`'
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBlock(channels, inner_channels, kernel_size=1,
                               dilation=dilations[0], norm_type=norm_type, act_type=act_type)
        self.aspp1 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[1], norm_type=norm_type, act_type=act_type)
        self.aspp2 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[2], norm_type=norm_type, act_type=act_type)
        self.aspp3 = ConvBlock(channels, inner_channels, kernel_size=3,
                               dilation=dilations[3], norm_type=norm_type)
        self.avg_conv = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                      ConvBlock(channels, inner_channels, kernel_size=1,
                                                dilation=1, norm_type=norm_type, act_type=act_type))
        self.transition = ConvBlock(cat_channels, channels, kernel_size=1,
                                    dilation=dilations[0], norm_type=norm_type, act_type=act_type)

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        avg = self.avg_conv(input)
        avg = F.interpolate(avg, aspp2.size()[2:], mode='nearest')
        out = torch.cat((aspp0, aspp1, aspp2, aspp3, avg), dim=1)
        out = self.transition(out)
        return out

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

class FPN3D(nn.Module):
    def __init__(self, n_filters, feature_size=64, norm_type='batchnorm', act_type='ReLU'):
        super(FPN3D, self).__init__()

        # FPN
        self.P5_1 = ConvBlock(n_filters[-1], feature_size, kernel_size=1, norm_type=norm_type, act_type='none')
        self.P5_upsampled = UpsamplingDeconvBlock(feature_size, feature_size, stride=2, norm_type=norm_type, act_type='none')
        # self.P5_2 = ConvBlock(feature_size, feature_size, kernel_size=3, norm_type=norm_type, act_type='none')
        
        self.P4_1 = ConvBlock(n_filters[-2], feature_size, kernel_size=1, norm_type=norm_type, act_type='none')
        self.P4_upsampled = UpsamplingDeconvBlock(feature_size, feature_size, stride=2, norm_type=norm_type, act_type='none')
        # self.P4_2 = ConvBlock(feature_size, feature_size, kernel_size=3, norm_type=norm_type, act_type='none')

        self.P3_1 = ConvBlock(n_filters[-3], feature_size, kernel_size=1, norm_type=norm_type, act_type='none')
        if len(n_filters) == 4:
            self.P3_upsampled = UpsamplingDeconvBlock(feature_size, feature_size, stride=2, norm_type=norm_type, act_type='none')
            # self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P2_1 = ConvBlock(n_filters[-4], feature_size, kernel_size=1, norm_type=norm_type, act_type='none')
            self.P2_2 = ConvBlock(feature_size, feature_size, kernel_size=3, norm_type=norm_type, act_type='none')
        else:
            self.P3_2 = ConvBlock(feature_size, feature_size, kernel_size=3, norm_type=norm_type, act_type='none')

    def forward(self, inputs):
        if getattr(self, 'P3_upsampled', None) is not None:
            C2, C3, C4, C5 = inputs
        else:
            C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        if hasattr(self, 'P3_upsampled'):
            P3_upsampled_x = self.P3_upsampled(P3_x)
            P2_x = self.P2_1(C2)
            P2_x = P2_x + P3_upsampled_x
            P2_x = self.P2_2(P2_x)
            return P2_x
        else:
            P3_x = self.P3_2(P3_x)
            return P3_x
        
class Resnet18(nn.Module):
    def __init__(self, n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,
                 norm_type='batchnorm', head_norm='batchnorm', act_type='ReLU', se=True, aspp=False, dw_type='conv', up_type='deconv', dropout=0.0,
                 first_stride=(2, 2, 2), detection_loss=None, device=None, out_stride=4):
        super(Resnet18, self).__init__()
        assert len(n_blocks) == 4, 'The length of n_blocks should be 4'
        assert len(n_filters) == 4, 'The length of n_filters should be 4'
        self.detection_loss = detection_loss
        self.device = device
        self.out_stride = out_stride
        # Stem
        # self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_conv = nn.Sequential(StemBlock(n_channels, stem_filters, norm_type=norm_type, act_type=act_type))
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=first_stride, norm_type=norm_type, act_type=act_type)
        
        # Encoder
        self.block1 = ELAN_Layer(n_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
        
        dw_block = DownsamplingConvBlock if dw_type == 'conv' else DownsamplingBlock
        self.block1_dw = dw_block(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = ELAN_Layer(n_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block2_dw = dw_block(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = ELAN_Layer(n_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block3_dw = dw_block(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        if aspp:
            self.block4 = ASPP(n_filters[3], norm_type=norm_type, act_type=act_type)
        else:
            self.block4 = ELAN_Layer(n_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type, se=se)

        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None
            
        if out_stride == 4:
            self.fpn = FPN3D(n_filters[1:], feature_size=n_filters[1], norm_type=norm_type, act_type=act_type)
            # Head
            self.head = ClsRegHead(in_channels=n_filters[1], feature_size=n_filters[1], conv_num=3, norm_type=head_norm, act_type=act_type)
        elif out_stride == 2:
            self.fpn = FPN3D(n_filters, feature_size=n_filters[0], norm_type=norm_type, act_type=act_type)
            # Head
            self.head = ClsRegHead(in_channels=n_filters[0], feature_size=n_filters[0], conv_num=3, norm_type=head_norm, act_type=act_type)
        
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
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x3 = self.block3(x)

        x = self.block3_dw(x3)
        if self.dropout is not None:
            x3 = self.dropout(x3)
        x4 = self.block4(x)

        if self.out_stride == 4:
            feats = self.fpn([x2, x3, x4])
        else:
            feats = self.fpn([x1, x2, x3, x4])
            
        "decode"
        out = self.head(feats)
        if self.training:
            cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = self.detection_loss(out, labels, device=self.device)
            return cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        prior = 0.01
        nn.init.constant_(self.head.cls_output.weight, 0)
        nn.init.constant_(self.head.cls_output.bias, -math.log((1.0 - prior) / prior))

        nn.init.constant_(self.head.shape_output.weight, 0)
        nn.init.constant_(self.head.shape_output.bias, 0.5)

        nn.init.constant_(self.head.offset_output.weight, 0)
        nn.init.constant_(self.head.offset_output.bias, 0.05)

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
    strides = torch.tensor([input_size[0] / d, input_size[1] / h, input_size[2] / w], dtype=dtype, device=device)
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
                 cls_num_neg = 10000,
                 cls_num_hard = 100,
                 cls_fn_weight = 4.0,
                 cls_fn_threshold = 0.8,
                 cls_neg_pos_ratio = 100,
                 cls_hard_fp_weight = 2.0,
                 cls_hard_fp_threshold = 0.7):
        super(DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        
        self.cls_num_neg = cls_num_neg
        self.cls_num_hard = cls_num_hard
        self.cls_fn_weight = cls_fn_weight
        self.cls_fn_threshold = cls_fn_threshold
        
        self.cls_neg_pos_ratio = cls_neg_pos_ratio
        self.cls_hard_fp_weight = cls_hard_fp_weight
        self.cls_hard_fp_threshold = cls_hard_fp_threshold
        
    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, 
                 neg_pos_ratio = 100, fn_weight = 4.0, fn_threshold = 0.8, hard_fp_weight = 2.0, hard_fp_threshold = 0.7):
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
        # classification_losses = []
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
                if hard_fp_threshold != -1 and hard_fp_weight != -1:
                    hard_FP_thre1, hard_FP_thrs2 = 0.5, 0.7
                    hard_FP_w1, hard_FP_w2 = 1.5, 2
                    hard_FP_weight = hard_FP_w1 + torch.clamp((cls_prob - hard_FP_thre1) / (hard_FP_thrs2 - hard_FP_thre1), min=0.0, max=1.0) * (hard_FP_w2 - hard_FP_w1)
                    hard_fp_index = torch.gt(cls_prob, hard_FP_w1) & (record_targets == 0)
                    cls_loss[hard_fp_index == 1] *= hard_FP_weight[hard_fp_index == 1]
                    # hard_FP_index = torch.gt(cls_prob, hard_fp_threshold) & (record_targets == 0)
                    # cls_loss[hard_FP_index == 1] *= hard_fp_weight
                    # less_hard_FP_index = torch.gt(cls_prob, 0.5) & (record_targets == 0) & torch.lt(cls_prob, hard_fp_threshold)
                    # cls_loss[less_hard_FP_index] *= 1.5
                    
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
                if hard_fp_threshold != -1 and hard_fp_weight != -1:
                    hard_FP_thre1, hard_FP_thrs2 = 0.5, 0.7
                    hard_FP_w1, hard_FP_w2 = 1.5, 2.0
                    hard_FP_weight = hard_FP_w1 + torch.clamp((cls_prob - hard_FP_thre1) / (hard_FP_thrs2 - hard_FP_thre1), min=0.0, max=1.0) * (hard_FP_w2 - hard_FP_w1)
                    hard_fp_index = torch.gt(cls_prob, hard_FP_w1) & (record_targets == 0)
                    cls_loss[hard_fp_index == 1] *= hard_FP_weight[hard_fp_index == 1]
                    # hard_FP_index = torch.gt(cls_prob, hard_fp_threshold) & (record_targets == 0)
                    # cls_loss[hard_FP_index == 1] *= hard_fp_weight
                    # less_hard_FP_index = torch.gt(cls_prob, 0.5) & (record_targets == 0) & torch.lt(cls_prob, hard_fp_threshold)
                    # cls_loss[less_hard_FP_index] *= 1.5
                
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
    def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
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
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
        mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
        del target_ctr, distance, mask_topk
        return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
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
        target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
                                                                                                     anchor_points = anchor_points,
                                                                                                     stride = stride_tensor[0].view(1, 1, 3), 
                                                                                                     pos_target_topk = self.pos_target_topk,
                                                                                                     ignore_ratio = self.pos_ignore_ratio)
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        mask_ignore = mask_ignore.int()
        cls_pos_loss, cls_neg_loss = self.cls_loss(pred = pred_scores, 
                                                   target = target_scores, 
                                                   mask_ignore = mask_ignore, 
                                                   neg_pos_ratio = self.cls_neg_pos_ratio,
                                                   num_hard = self.cls_num_hard, 
                                                   num_neg = self.cls_num_neg,
                                                   fn_weight = self.cls_fn_weight, 
                                                   fn_threshold = self.cls_fn_threshold,
                                                   hard_fp_weight = self.cls_hard_fp_weight,
                                                   hard_fp_threshold = self.cls_hard_fp_threshold)
        
        # Only calculate the loss of positive samples                                 
        fg_mask = target_scores.squeeze(-1).bool()
        if fg_mask.sum() == 0:
            reg_loss = torch.tensor(0.0, device=device)
            offset_loss = torch.tensor(0.0, device=device)
            iou_loss = torch.tensor(0.0, device=device)
        else:
            reg_loss = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
            offset_loss = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            iou_loss = 1 - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
        return cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss

class DetectionPostprocess(nn.Module):
    def __init__(self, topk: int=60, threshold: float=0.15, nms_threshold: float=0.05, nms_topk: int=20, crop_size: List[int]=[96, 96, 96], min_size: int=-1):
        super(DetectionPostprocess, self).__init__()
        self.topk = topk
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = nms_topk
        self.crop_size = crop_size
        self.min_size = min_size

    def forward(self, output, device, is_logits=True):
        Cls = output['Cls']
        Shape = output['Shape']
        Offset = output['Offset']
        batch_size = Cls.size()[0]
        anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0)
        # view shape
        pred_scores = Cls.view(batch_size, 1, -1)
        pred_shapes = Shape.view(batch_size, 3, -1)
        pred_offsets = Offset.view(batch_size, 3, -1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        if is_logits:
            pred_scores = pred_scores.sigmoid()
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
        
        if self.min_size > 0:
            dets_volumes = dets[:, :, 4] * dets[:, :, 5] * dets[:, :, 6]
            dets[dets_volumes < self.min_size] = -1
                
        return dets