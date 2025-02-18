import re
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

class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(data=init_val), requires_grad=True)

    def forward(self, x):
        return x * self.scale

class SubNetFC(nn.Module):
    def __init__(self, m_top_k = 4, inner_channel = 64, add_mean = True):
        super(SubNetFC, self).__init__()
        self.m_top_k = m_top_k
        self.add_mean = add_mean
        total_dim = (m_top_k + 1) if add_mean else m_top_k
        self.reg_conf = nn.Sequential(nn.Conv3d(total_dim * 3, inner_channel, kernel_size=1),
                                    nn.ReLU(inplace = True),
                                    nn.Conv3d(inner_channel, 1, kernel_size=1))
    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                The input tensor of shape (bs, 3*(reg_max + 1), d, h, w).
        """
        bs, n, d, h, w = x.shape
        x = x.view(x.shape[0], 3, -1, d, h, w) # [bs, 3, reg_max+1, d, h, w]
        x = F.softmax(x, dim = 2) # [bs, 3, reg_max+1, d, h, w]
        prob_topk, _ = x.topk(self.m_top_k, dim = 2)  # shape=[bs, 3, m_top_k, d, h, w]
        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim = 2, keepdim = True)], dim=2) # shape=[bs, 3, m_top_k+1, d, h, w]
        else:
            stat = prob_topk # shape=[bs, 3, m_top_k, d, h, w]
        quality_score = self.reg_conf(stat.view(bs, -1, d, h, w)) # shape=[bs, 1, d, h, w]
        return quality_score
    
class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.

    From generalized focal loss v2

    """

    def __init__(self, reg_min=0., reg_max=40., bins=41):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.reg_min = reg_min
        self.register_buffer(
            "project", torch.linspace(self.reg_min, self.reg_max, bins)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
        Returns:
            x (Tensor): Integral result of box locations, i.e., nodule dhw.
        """
        bs, n, d, h, w = x.shape
        num_points = d * h * w
        x = x.view(bs, n, num_points).permute(0, 2, 1).contiguous().view(bs, num_points, 3, -1) # [bs, z*y*x, 3, bins]
        x = F.softmax(x, dim = -1) # [bs, z*y*x, 3, bins]
        x = F.linear(x, self.project.type_as(x)) # [bs, z*y*x, 3]
        x = x.permute(0, 2, 1).contiguous() # [bs, 3, z*y*x]
        return x

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
    def __init__(self, channels, ratio=4,
                 dilations=[1, 2, 3, 4],
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
                 norm_type='groupnorm', act_type='LeakyReLU', reg_max=40):
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
        self.shape_output = nn.Conv3d(feature_size, 3 * (reg_max + 1), kernel_size=3, padding=1)
        self.offset_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)
        
        self.shape_intergral = Integral(reg_max=reg_max, bins=reg_max+1)
        self.offset_intergral = Integral(reg_min=-2.5, reg_max=2.5, bins=reg_max+1)
        self.shape_scale = Scale()
        self.offset_scale = Scale()
        self.subnet = SubNetFC(m_top_k = 4, inner_channel = 64, add_mean = True)
        
    def forward(self, x):
        Shape_feat = self.shape_scale(self.shape_output(self.conv_r(x))) # [bs, 3*(reg_max+1), d, h, w]
        Offset_feat = self.offset_scale(self.offset_output(self.conv_o(x))) # [bs, 3*(reg_max+1), d, h, w]
        # Offset = self.offset_output(self.conv_o(x))
        Cls = self.cls_output(self.conv_s(x)) # shape = [bs, 1, d, h, w]
        dict1 = {}
        
        # cls_weight = self.subnet(Shape_feat).sigmoid() # [bs, d*h*w]
        # cls_weight = cls_weight.view(bs, 1, d, h, w)
        dict1['Cls'] = Cls * self.subnet(Shape_feat).sigmoid() # [bs, 1, d, h, w]
        dict1['Shape'] = self.shape_intergral(Shape_feat)
        dict1['Offset'] = self.offset_intergral(Offset_feat)  
        
        bs, n, d, h, w = Shape_feat.shape
        num_points = d * h * w
        # dict1['Shape_feat'] = Shape_feat.view(bs, n, num_points).permute(0, 2, 1).contiguous().view(bs, num_points, 3, -1) # [bs, z*y*x, 3, reg_max+1]
        dict1['Shape_feat'] = Shape_feat.view(bs, -1, num_points).permute(0, 2, 1).contiguous() # [bs, z*y*x, 3, reg_max+1]
        dict1['Offset_feat'] = Offset_feat.view(bs, -1, num_points).permute(0, 2, 1).contiguous() # [bs, z*y*x, 3, reg_max+1]
        return dict1

class Resnet18(nn.Module):
    def __init__(self, n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,
                 norm_type='batchnorm', head_norm='batchnorm', act_type='ReLU', se=True, aspp=False, dw_type='conv', up_type='deconv', dropout=0.0,
                 first_stride=(2, 2, 2), detection_loss=None, device=None, reg_max=40, out_stride=4):
        super(Resnet18, self).__init__()
        assert len(n_blocks) == 4, 'The length of n_blocks should be 4'
        assert len(n_filters) == 4, 'The length of n_filters should be 4'
        self.detection_loss = detection_loss
        self.device = device

        # Stem
        self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=first_stride, norm_type=norm_type, act_type=act_type)
        
        # Encoder
        self.block1 = LayerBasic(n_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
        
        dw_block = DownsamplingConvBlock if dw_type == 'conv' else DownsamplingBlock
        self.block1_dw = dw_block(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = LayerBasic(n_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block2_dw = dw_block(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = LayerBasic(n_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block3_dw = dw_block(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        if aspp:
            self.block4 = ASPP(n_filters[3], norm_type=norm_type, act_type=act_type)
        else:
            self.block4 = LayerBasic(n_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type, se=se)

        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None
            
        # Decoder
        up_block = UpsamplingDeconvBlock if up_type == 'deconv' else UpsamplingBlock
        self.block33_up = up_block(n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_res = LayerBasic(1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se)
        self.block33 = LayerBasic(2, n_filters[2] * 2, n_filters[2], norm_type=norm_type, act_type=act_type, se=se)

        self.block22_up = up_block(n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_res = LayerBasic(1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        self.block22 = LayerBasic(2, n_filters[1] * 2, n_filters[1], norm_type=norm_type, act_type=act_type, se=se)
        
        # Head
        self.head = ClsRegHead(in_channels=n_filters[1], feature_size=n_filters[1], conv_num=3, norm_type=head_norm, act_type=act_type, reg_max = reg_max)
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

        x = self.block4(x)

        "decode"
        x = self.block33_up(x)
        x3 = self.block33_res(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)

        out = self.head(x)
        if self.training:
            cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss = self.detection_loss(out, labels, device=self.device)
            return cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss
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
        nn.init.constant_(self.head.shape_output.bias, -math.log((1.0 - prior) / prior))

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
                 cls_num_hard = 100,
                 cls_fn_weight = 4.0,
                 cls_fn_threshold = 0.8,
                 reg_max = 40):
        super(DetectionLoss, self).__init__()
        self.crop_size = crop_size
        self.pos_target_topk = pos_target_topk
        self.pos_ignore_ratio = pos_ignore_ratio
        
        self.cls_num_hard = cls_num_hard
        self.cls_fn_weight = cls_fn_weight
        self.cls_fn_threshold = cls_fn_threshold
        self.reg_max = reg_max
        
    @staticmethod  
    def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, ratio = 100, fn_weight = 4.0, fn_threshold = 0.8):
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
                # FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(cls_prob, fn_threshold) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] = fn_weight * cls_loss[FN_index == 1]
                
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
                       ignore_ratio = 3,
                       reg_max = 40):# larger the ignore_ratio, the more GPU memory is used
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
        target_shape = torch.clamp(target_shape, min=None, max=reg_max) # restrict the shape to reg_max, first 0.5 is mean the half of the shape
        
        target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
        target_bboxes[:, 3:] = torch.clamp(target_bboxes[:, 3:], min=None, max=reg_max * 2) # restrict the shape to reg_max
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
        Cls = output['Cls'] # # shape = [bs, 1, z, y, x]
        Shape = output['Shape']
        Offset = output['Offset']
        Shape_feat = output['Shape_feat'] # [bs, num_points, 3, reg_max+1]
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
                                                                                                     ignore_ratio = self.pos_ignore_ratio,
                                                                                                     reg_max=self.reg_max)
        # merge mask ignore
        mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
        fg_mask = target_scores.squeeze(-1).bool() # shape = (b, num_points)
        classification_losses = self.cls_loss(pred_scores, target_scores, mask_ignore.int(), num_hard=self.cls_num_hard, fn_weight=self.cls_fn_weight, fn_threshold=self.cls_fn_threshold)
        if fg_mask.sum() == 0:
            reg_losses = torch.tensor(0).float().to(device)
            offset_losses = torch.tensor(0).float().to(device)
            iou_losses = torch.tensor(0).float().to(device)
        else:
            reg_losses = self.distribution_focal_loss(Shape_feat[fg_mask].view(-1, self.reg_max+1), target_shape[fg_mask].view(-1))
            offset_losses = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
            # print('ppred:', pred_bboxes[fg_mask][:5])
            # print('gt:', target_bboxes[fg_mask][:5])
            iou_losses = -(self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
            
        return classification_losses, reg_losses, offset_losses, iou_losses

    @staticmethod
    def distribution_focal_loss(pred, label):
        r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
        <https://arxiv.org/abs/2006.04388>`_.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding boxes
                (before softmax) with shape (N, n+1), n is the max value of the
                integral set `{0, ..., n}` in paper.
            label (torch.Tensor): Target distance label for bounding boxes with
                shape (N,).

        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """
        dis_left = label.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - label
        weight_right = label - dis_left.float()
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
            + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
        return loss

    
class DetectionPostprocess(nn.Module):
    def __init__(self, topk: int=60, threshold: float=0.15, nms_threshold: float=0.05, nms_topk: int=20, crop_size: List[int]=[96, 96, 96]):
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