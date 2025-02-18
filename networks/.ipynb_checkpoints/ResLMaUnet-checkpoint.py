from typing import Tuple, List, Union, Dict
import random
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from .modules import SELayer, Identity, ConvBlock, act_layer, norm_layer3d

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
        
        if out_stride == 4:
            # Head
            self.head = ClsRegHead(in_channels=n_filters[1], feature_size=n_filters[1], conv_num=3, norm_type=head_norm, act_type=act_type)
        elif out_stride == 2:
            self.block11_up = up_block(n_filters[1], n_filters[0], norm_type=norm_type, act_type=act_type)
            self.block11_res = LayerBasic(1, n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
            self.block11 = LayerBasic(2, n_filters[0] * 2, n_filters[0], norm_type=norm_type, act_type=act_type, se=se)
            # Head
            self.head = ClsRegHead(in_channels=n_filters[0], feature_size=n_filters[0], conv_num=3, norm_type=head_norm, act_type=act_type)
        self._init_weight()

    def forward(self, inputs):
        if self.training and self.detection_loss != None:
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

        if hasattr(self, 'block11_up'):
            x = self.block11_up(x)
            x1 = self.block11_res(x1)
            x = torch.cat([x, x1], dim=1)
            x = self.block11(x)

        out = self.head(x)
        if self.training and self.detection_loss != None:
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
