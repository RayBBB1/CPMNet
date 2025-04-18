#2p,2w
from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

from re import S
from xml.dom import xmlbuilder
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from mamba_ssm import Mamba
import copy
import math
import sys
from .modules import SELayer, Identity, ConvBlock, act_layer, norm_layer3d
from ptflops import get_model_complexity_info
'''
Computational complexity:       93.05 GMac
Number of parameters:           12.5 M  
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [48, 24, 12, 6]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        # outs.shape = [[48*3,48c],[24*3,96c],[12*3,192c],[6*3,384c]])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)

                
        return dec2
        
class Resnet18(nn.Module):

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None,
                 detection_loss = None,
                 device = None
                 ):
        super().__init__()
        self.detection_loss = detection_loss
        self.device = device
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        # assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
        #                                           f"resolution stages. here: {n_stages}. " \
        #                                           f"n_blocks_per_stage: {n_blocks_per_stage}"
        # assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
        #                                                         f"as we have resolution stages. here: {n_stages} " \
        #                                                         f"stages, so it should have {n_stages - 1} entries. " \
        #                                                         f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.feature_extractor = SegMamba(in_chans=1,
                        out_chans=1,
                        depths=[2,3,3,3],
                        feat_size= features_per_stage)
                     
        self.head =  ClsRegHead(in_channels=features_per_stage[1], feature_size=features_per_stage[1], conv_num=3, norm_type="batchnorm", act_type="ReLU")
    def forward(self, inputs):
        
        if self.training and self.detection_loss != None:
            x, labels = inputs
        else:
            x = inputs
        out_feature = self.feature_extractor(x)
        out = self.head(out_feature)
        if self.training and self.detection_loss != None:
            cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss = self.detection_loss(out, labels, device=self.device)
            return cls_pos_loss, cls_neg_loss, shape_loss, offset_loss, iou_loss
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


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

# from typing import Tuple, List, Union, Dict
# import random
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# import numpy as np
# from utils.box_utils import bbox_decode, make_anchors, zyxdhw2zyxzyx

# class DetectionLoss(nn.Module):
#     def __init__(self, 
#                  crop_size=[64, 128, 128], 
#                  pos_target_topk = 7, 
#                  pos_ignore_ratio = 3,
#                  cls_num_neg = 10000,
#                  cls_num_hard = 100,
#                  cls_fn_weight = 4.0,
#                  cls_fn_threshold = 0.8,
#                  cls_neg_pos_ratio = 100,
#                  cls_hard_fp_thrs1 = 0.5,
#                  cls_hard_fp_thrs2 = 0.7,
#                  cls_hard_fp_w1 = 1.5,
#                  cls_hard_fp_w2 = 2.0):
#         super(DetectionLoss, self).__init__()
#         self.crop_size = crop_size
#         self.pos_target_topk = pos_target_topk
#         self.pos_ignore_ratio = pos_ignore_ratio
        
#         self.cls_num_neg = cls_num_neg
#         self.cls_num_hard = cls_num_hard
#         self.cls_fn_weight = cls_fn_weight
#         self.cls_fn_threshold = cls_fn_threshold
        
#         self.cls_neg_pos_ratio = cls_neg_pos_ratio
        
#         self.cls_hard_fp_thrs1 = cls_hard_fp_thrs1
#         self.cls_hard_fp_thrs2 = cls_hard_fp_thrs2
#         self.cls_hard_fp_w1 = cls_hard_fp_w1
#         self.cls_hard_fp_w2 = cls_hard_fp_w2
        
#     @staticmethod  
#     def cls_loss(pred: torch.Tensor, target, mask_ignore, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, neg_pos_ratio = 100, fn_weight = 4.0, fn_threshold = 0.8, 
#                  hard_fp_thrs1 = 0.5, hard_fp_thrs2 = 0.7, hard_fp_w1 = 1.5, hard_fp_w2 = 2.0):
#         """
#         Calculates the classification loss using focal loss and binary cross entropy.

#         Args:
#             pred (torch.Tensor): The predicted logits of shape (b, num_points, 1)
#             target: The target labels of shape (b, num_points, 1)
#             mask_ignore: The mask indicating which pixels to ignore of shape (b, num_points, 1)
#             alpha (float): The alpha factor for focal loss (default: 0.75)
#             gamma (float): The gamma factor for focal loss (default: 2.0)
#             num_neg (int): The maximum number of negative pixels to consider (default: 10000, if -1, use all negative pixels)
#             num_hard (int): The number of hard negative pixels to keep (default: 100)
#             ratio (int): The ratio of negative to positive pixels to consider (default: 100)
#             fn_weight (float): The weight for false negative pixels (default: 4.0)
#             fn_threshold (float): The threshold for considering a pixel as a false negative (default: 0.8)
#             hard_fp_weight (float): The weight for hard false positive pixels (default: 2.0)
#             hard_fp_threshold (float): The threshold for considering a pixel as a hard false positive (default: 0.7)
#         Returns:
#             torch.Tensor: The calculated classification loss
#         """
#         cls_pos_losses = []
#         cls_neg_losses = []
#         batch_size = pred.shape[0]
#         for j in range(batch_size):
#             pred_b = pred[j]
#             target_b = target[j]
#             mask_ignore_b = mask_ignore[j]
            
#             # Calculate the focal weight
#             cls_prob = torch.sigmoid(pred_b.detach())
#             cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
#             alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * alpha
#             alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
#             focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
#             focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

#             # Calculate the binary cross entropy loss
#             bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
#             num_positive_pixels = torch.sum(target_b == 1)
#             cls_loss = focal_weight * bce
#             cls_loss = torch.where(torch.eq(mask_ignore_b, 0), cls_loss, 0)
#             record_targets = target_b.clone()
#             if num_positive_pixels > 0:
#                 # Weight the hard false negatives(FN)
#                 FN_index = torch.lt(cls_prob, fn_threshold) & (record_targets == 1)  # 0.9
#                 cls_loss[FN_index == 1] *= fn_weight
                
#                 # Weight the hard false positives(FP)
#                 if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
#                     hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
#                     hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
#                     cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                    
#                 Positive_loss = cls_loss[record_targets == 1]
#                 Negative_loss = cls_loss[record_targets == 0]
#                 # Randomly sample negative pixels
#                 if num_neg != -1:
#                     neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss))) 
#                     Negative_loss = Negative_loss[neg_idcs]
                    
#                 # Get the top k negative pixels
#                 _, keep_idx = torch.topk(Negative_loss, min(neg_pos_ratio * num_positive_pixels, len(Negative_loss))) 
#                 Negative_loss = Negative_loss[keep_idx] 
                
#                 # Calculate the loss
#                 num_positive_pixels = torch.clamp(num_positive_pixels.float(), min=1.0)
#                 Positive_loss = Positive_loss.sum() / num_positive_pixels
#                 Negative_loss = Negative_loss.sum() / num_positive_pixels
#                 cls_pos_losses.append(Positive_loss)
#                 cls_neg_losses.append(Negative_loss)
#             else: # no positive pixels
#                 # Weight the hard false positives(FP)
#                 if hard_fp_thrs1 != -1 and hard_fp_w1 != -1 and hard_fp_thrs2 != -1 and hard_fp_w2 != -1:
#                     hard_FP_weight = hard_fp_w1 + torch.clamp((cls_prob - hard_fp_thrs1) / (hard_fp_thrs2 - hard_fp_thrs1), min=0.0, max=1.0) * (hard_fp_w2 - hard_fp_w1)
#                     hard_FP_index = torch.gt(cls_prob, hard_fp_thrs1) & (record_targets == 0)
#                     cls_loss[hard_FP_index == 1] *= hard_FP_weight[hard_FP_index == 1]
                
#                 # Randomly sample negative pixels
#                 Negative_loss = cls_loss[record_targets == 0]
#                 if num_neg != -1:
#                     neg_idcs = random.sample(range(len(Negative_loss)), min(num_neg, len(Negative_loss)))
#                     Negative_loss = Negative_loss[neg_idcs]
                
#                 # Get the top k negative pixels   
#                 _, keep_idx = torch.topk(Negative_loss, num_hard)
                
#                 # Calculate the loss
#                 Negative_loss = Negative_loss[keep_idx]
#                 Negative_loss = Negative_loss.sum()
#                 cls_neg_losses.append(Negative_loss)
                
#         if len(cls_pos_losses) == 0:
#             cls_pos_loss = torch.tensor(0.0, device=pred.device)
#         else:
#             cls_pos_loss = torch.sum(torch.stack(cls_pos_losses)) / batch_size
            
#         if len(cls_neg_losses) == 0:
#             cls_neg_loss = torch.tensor(0.0, device=pred.device)
#         else:
#             cls_neg_loss = torch.sum(torch.stack(cls_neg_losses)) / batch_size
#         return cls_pos_loss, cls_neg_loss
    
#     @staticmethod
#     def target_proprocess(annotations: torch.Tensor, 
#                           device, 
#                           input_size: List[int],
#                           mask_ignore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Preprocess the annotations to generate the targets for the network.
#         In this function, we remove some annotations that the area of nodule is too small in the crop box. (Probably cropped by the edge of the image)
        
#         Args:
#             annotations: torch.Tensor
#                 A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
#                 (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
#             device: torch.device, the device of the model.
#             input_size: List[int]
#                 A list of length 3 containing the (z, y, x) dimensions of the input.
#             mask_ignore: torch.Tensor
#                 A zero tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
#         Returns: 
#             A tuple of two tensors:
#                 (1) annotations_new: torch.Tensor
#                     A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
#                     (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
#                 (2) mask_ignore: torch.Tensor
#                     A tensor of shape (batch_size, 1, z, y, x) to store the mask ignore.
#         """
#         batch_size = annotations.shape[0]
#         annotations_new = -1 * torch.ones_like(annotations, device=device)
#         for sample_i in range(batch_size):
#             annots = annotations[sample_i]
#             gt_bboxes = annots[annots[:, -1] > -1] # -1 means ignore, it is used to make each sample has same number of bbox (pad with -1)
#             bbox_annotation_target = []
            
#             crop_box = torch.tensor([0., 0., 0., input_size[0], input_size[1], input_size[2]], device=device)
#             for s in range(len(gt_bboxes)):
#                 each_label = gt_bboxes[s] # (z_ctr, y_ctr, x_ctr, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1)
#                 # coordinate convert zmin, ymin, xmin, d, h, w
#                 z1 = (torch.max(each_label[0] - each_label[3]/2., crop_box[0]))
#                 y1 = (torch.max(each_label[1] - each_label[4]/2., crop_box[1]))
#                 x1 = (torch.max(each_label[2] - each_label[5]/2., crop_box[2]))

#                 z2 = (torch.min(each_label[0] + each_label[3]/2., crop_box[3]))
#                 y2 = (torch.min(each_label[1] + each_label[4]/2., crop_box[4]))
#                 x2 = (torch.min(each_label[2] + each_label[5]/2., crop_box[5]))
                
#                 nd = torch.clamp(z2 - z1, min=0.0)
#                 nh = torch.clamp(y2 - y1, min=0.0)
#                 nw = torch.clamp(x2 - x1, min=0.0)
#                 if nd * nh * nw == 0:
#                     continue
#                 percent = nw * nh * nd / (each_label[3] * each_label[4] * each_label[5])
#                 if (percent > 0.1) and (nw*nh*nd >= 15):
#                     spacing_z, spacing_y, spacing_x = each_label[6:9]
#                     bbox = torch.from_numpy(np.array([float(z1 + 0.5 * nd), float(y1 + 0.5 * nh), float(x1 + 0.5 * nw), float(nd), float(nh), float(nw), float(spacing_z), float(spacing_y), float(spacing_x), 0])).to(device)
#                     bbox_annotation_target.append(bbox.view(1, 10))
#                 else:
#                     mask_ignore[sample_i, 0, int(z1) : int(torch.ceil(z2)), int(y1) : int(torch.ceil(y2)), int(x1) : int(torch.ceil(x2))] = -1
#             if len(bbox_annotation_target) > 0:
#                 bbox_annotation_target = torch.cat(bbox_annotation_target, 0)
#                 annotations_new[sample_i, :len(bbox_annotation_target)] = bbox_annotation_target
#         return annotations_new, mask_ignore
    
#     @staticmethod
#     def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
#         box1 = zyxdhw2zyxzyx(box1)
#         box2 = zyxdhw2zyxzyx(box2)
#         # Get the coordinates of bounding boxes
#         b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
#         b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
#         w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
#         w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

#         # Intersection area
#         inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
#                 (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
#                 (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

#         # Union Area
#         union = w1 * h1 * d1 + w2 * h2 * d2 - inter

#         # IoU
#         iou = inter / union
#         if DIoU:
#             cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#             ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#             cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
#             c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
#             + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
#             return iou - rho2 / c2  # DIoU
#         return iou  # IoU
    
#     @staticmethod
#     def get_pos_target(annotations: torch.Tensor,
#                        anchor_points: torch.Tensor,
#                        stride: torch.Tensor,
#                        pos_target_topk = 7, 
#                        ignore_ratio = 3):# larger the ignore_ratio, the more GPU memory is used
#         """Get the positive targets for the network.
#         Steps:
#             1. Calculate the distance between each annotation and each anchor point.
#             2. Find the top k anchor points with the smallest distance for each annotation.
#         Args:
#             annotations: torch.Tensor
#                 A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format: (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1).
#             anchor_points: torch.Tensor
#                 A tensor of shape (num_of_point, 3) containing the coordinates of the anchor points, each of which is in the format (z, y, x).
#             stride: torch.Tensor
#                 A tensor of shape (1,1,3) containing the strides of each dimension in format (z, y, x).
#         """
#         batchsize, num_of_annots, _ = annotations.size()
#         # -1 means ignore, larger than 0 means positive
#         # After the following operation, mask_gt will be a tensor of shape (batch_size, num_annotations, 1) 
#         # indicating whether the annotation is ignored or not.
#         mask_gt = annotations[:, :, -1].clone().gt_(-1) # (b, num_annotations)
        
#         # The coordinates in annotations is on original image, we need to convert it to the coordinates on the feature map.
#         ctr_gt_boxes = annotations[:, :, :3] / stride # z0, y0, x0
#         shape = annotations[:, :, 3:6] / 2 # half d h w
        
#         sp = annotations[:, :, 6:9] # spacing, shape = (b, num_annotations, 3)
#         sp = sp.unsqueeze(-2) # shape = (b, num_annotations, 1, 3)
        
#         # distance (b, n_max_object, anchors)
#         distance = -(((ctr_gt_boxes.unsqueeze(2) - anchor_points.unsqueeze(0)) * sp).pow(2).sum(-1))
#         _, topk_inds = torch.topk(distance, (ignore_ratio + 1) * pos_target_topk, dim=-1, largest=True, sorted=True)
        
#         mask_topk = F.one_hot(topk_inds[:, :, :pos_target_topk], distance.size()[-1]).sum(-2) # (b, num_annotation, num_of_points), the value is 1 or 0
#         mask_ignore = -1 * F.one_hot(topk_inds[:, :, pos_target_topk:], distance.size()[-1]).sum(-2) # the value is -1 or 0
        
#         # the value is 1 or 0, shape= (b, num_annotations, num_of_points)
#         # mask_gt is 1 mean the annotation is not ignored, 0 means the annotation is ignored
#         # mask_topk is 1 means the point is assigned to positive
#         # mask_topk * mask_gt.unsqueeze(-1) is 1 means the point is assigned to positive and the annotation is not ignored
#         mask_pos = mask_topk * mask_gt.unsqueeze(-1) 
        
#         mask_ignore = mask_ignore * mask_gt.unsqueeze(-1) # the value is -1 or 0, shape= (b, num_annotations, num_of_points)
#         gt_idx = mask_pos.argmax(-2) # shape = (b, num_of_points), it indicates each point matches which annotation
        
#         # Flatten the batch dimension
#         batch_ind = torch.arange(end=batchsize, dtype=torch.int64, device=ctr_gt_boxes.device)[..., None] # (b, 1)
#         gt_idx = gt_idx + batch_ind * num_of_annots
        
#         # Generate the targets of each points
#         target_ctr = ctr_gt_boxes.view(-1, 3)[gt_idx]
#         target_offset = target_ctr - anchor_points
#         target_shape = shape.view(-1, 3)[gt_idx]
        
#         target_bboxes = annotations[:, :, :6].view(-1, 6)[gt_idx] # zyxdhw
#         target_scores, _ = torch.max(mask_pos, 1) # shape = (b, num_of_points), the value is 1 or 0, 1 means the point is assigned to positive
#         mask_ignore, _ = torch.min(mask_ignore, 1) # shape = (b, num_of_points), the value is -1 or 0, -1 means the point is ignored
#         del target_ctr, distance, mask_topk
#         return target_offset, target_shape, target_bboxes, target_scores.unsqueeze(-1), mask_ignore.unsqueeze(-1)
    
#     def forward(self, 
#                 output: Dict[str, torch.Tensor], 
#                 annotations: torch.Tensor,
#                 device):
#         """
#         Args:
#             output: Dict[str, torch.Tensor], the output of the model.
#             annotations: torch.Tensor
#                 A tensor of shape (batch_size, num_annotations, 10) containing the annotations in the format:
#                 (ctr_z, ctr_y, ctr_x, d, h, w, spacing_z, spacing_y, spacing_x, 0 or -1). The last index -1 means the annotation is ignored.
#             device: torch.device, the device of the model.
#         """
#         Cls = output['Cls']
#         Shape = output['Shape']
#         Offset = output['Offset']
#         batch_size = Cls.size()[0]
#         target_mask_ignore = torch.zeros(Cls.size()).to(device)
        
#         # view shape
#         pred_scores = Cls.view(batch_size, 1, -1) # (b, 1, num_points)
#         pred_shapes = Shape.view(batch_size, 3, -1) # (b, 3, num_points)
#         pred_offsets = Offset.view(batch_size, 3, -1)
#         # (b, num_points, 1 or 3)
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_shapes = pred_shapes.permute(0, 2, 1).contiguous()
#         pred_offsets = pred_offsets.permute(0, 2, 1).contiguous()
        
#         # process annotations
#         process_annotations, target_mask_ignore = self.target_proprocess(annotations, device, self.crop_size, target_mask_ignore)
#         target_mask_ignore = target_mask_ignore.view(batch_size, 1,  -1)
#         target_mask_ignore = target_mask_ignore.permute(0, 2, 1).contiguous()
#         # generate center points. Only support single scale feature
#         anchor_points, stride_tensor = make_anchors(Cls, self.crop_size, 0) # shape = (num_anchors, 3)
#         # predict bboxes (zyxdhw)
#         pred_bboxes = bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor) # shape = (b, num_anchors, 6)
#         # assigned points and targets (target bboxes zyxdhw)
#         target_offset, target_shape, target_bboxes, target_scores, mask_ignore = self.get_pos_target(annotations = process_annotations,
#                                                                                                      anchor_points = anchor_points,
#                                                                                                      stride = stride_tensor[0].view(1, 1, 3), 
#                                                                                                      pos_target_topk = self.pos_target_topk,
#                                                                                                      ignore_ratio = self.pos_ignore_ratio)
#         # merge mask ignore
#         mask_ignore = mask_ignore.bool() | target_mask_ignore.bool()
#         mask_ignore = mask_ignore.int()
#         cls_pos_loss, cls_neg_loss = self.cls_loss(pred = pred_scores, 
#                                                    target = target_scores, 
#                                                    mask_ignore = mask_ignore, 
#                                                    neg_pos_ratio = self.cls_neg_pos_ratio,
#                                                    num_hard = self.cls_num_hard, 
#                                                    num_neg = self.cls_num_neg,
#                                                    fn_weight = self.cls_fn_weight, 
#                                                    fn_threshold = self.cls_fn_threshold,
#                                                    hard_fp_thrs1=self.cls_hard_fp_thrs1,
#                                                    hard_fp_thrs2=self.cls_hard_fp_thrs2,
#                                                     hard_fp_w1=self.cls_hard_fp_w1,
#                                                     hard_fp_w2=self.cls_hard_fp_w2)
        
#         # Only calculate the loss of positive samples                                 
#         fg_mask = target_scores.squeeze(-1).bool()
#         if fg_mask.sum() == 0:
#             reg_loss = torch.tensor(0.0, device=device)
#             offset_loss = torch.tensor(0.0, device=device)
#             iou_loss = torch.tensor(0.0, device=device)
#         else:
#             reg_loss = torch.abs(pred_shapes[fg_mask] - target_shape[fg_mask]).mean()
#             offset_loss = torch.abs(pred_offsets[fg_mask] - target_offset[fg_mask]).mean()
#             iou_loss = 1 - (self.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])).mean()
        
#         return cls_pos_loss, cls_neg_loss, reg_loss, offset_loss, iou_loss
# def main():
#     device = torch.device("cuda:0")
#     model = Resnet18(
#         input_channels=1,
#         n_stages=4,
#         features_per_stage=[64, 96, 128,160],
#         conv_op=nn.Conv3d,
#         kernel_sizes=[3, 3,3],
#         strides=[2,2,2],
#         n_conv_per_stage=[2, 2, 2,2],
#         num_classes=1,
#         n_conv_per_stage_decoder=[2],# conv after  transpose, not on skip connection
#         conv_bias=True,
#         norm_op=nn.BatchNorm3d,
#         norm_op_kwargs={'eps': 1e-5, 'affine': True},
#         dropout_op=None,
#         dropout_op_kwargs=None,
#         nonlin=nn.ReLU,
#         nonlin_kwargs={'inplace': True},
#         detection_loss = None,
#         device = device
#         ).to(device)
    
#     macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
#                                         print_per_layer_stat=True, verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# main()
