# from fvcore.nn import FlopCountAnalysis
import ipdb
import torch
import torch.nn as nn
from functools import partial
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from .modules import SELayer, Identity, ConvBlock, act_layer, norm_layer3d
from ptflops import get_model_complexity_info
from torch.cuda.amp import autocast
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
try:
    # from flopth import flopth
    from fvcore.nn import FlopCountAnalysis
except:
    pass
try:
    from VSS3D import VSSLayer3D
    from Transformer import TransformerBottleneck
except:
    from models.SS3D import SS3D, SS3D_v5, SS3D_v6
    from models.VSS3D import VSSLayer3D
    from models.Transformer import TransformerBottleneck
"""
Computational complexity:       164.17 GMac
Number of parameters:           43.23 M 
"""
class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class ResConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, id = True, preact = True): #id was False
        super().__init__()
        if preact:
            self.conv = nn.Sequential(
                nn.GroupNorm(8, ch_in),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                )
        id = (ch_in == ch_out) and id
        self.identity = (nn.Identity()) if id else nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        x = self.conv(inp)
        residual = self.identity(inp)
        return residual + x
def channel_shuffle_3d(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, depth, height, width = x.size()
    assert num_channels % groups == 0, f"num_channels ({num_channels}) must be divisible by groups ({groups})"
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, depth, height, width] -> [batch_size, groups, channels_per_group, depth, height, width]
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)

    # transpose groups and channels_per_group
    x = torch.transpose(x, 1, 2).contiguous()  # 交換第1維(groups)和第2維(channels_per_group)

    # flatten
    x = x.view(batch_size, num_channels, depth, height, width)

    return x  
class PatchMerging3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # Typically 2 * in_channels
        assert self.out_channels == 2 * self.in_channels, "Patch Merging typically doubles channels"

        # Conv layer for channel reduction
        self.reduction = nn.Conv3d(8 * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        B, C, D, H, W = x.shape

        # Ensure dimensions are even for downsampling
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Pad D, H, W dims to be even
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, D % 2)) # Pad last dim first (W), then H, then D

        # Step 2 & 3: Reshape and Concatenate
        x0 = x[:, :, 0::2, 0::2, 0::2]  # B, C, D/2, H/2, W/2
        x1 = x[:, :, 1::2, 0::2, 0::2]  # B, C, D/2, H/2, W/2
        x2 = x[:, :, 0::2, 1::2, 0::2]  # B, C, D/2, H/2, W/2
        x3 = x[:, :, 0::2, 0::2, 1::2]  # B, C, D/2, H/2, W/2
        x4 = x[:, :, 1::2, 1::2, 0::2]  # B, C, D/2, H/2, W/2
        x5 = x[:, :, 1::2, 0::2, 1::2]  # B, C, D/2, H/2, W/2
        x6 = x[:, :, 0::2, 1::2, 1::2]  # B, C, D/2, H/2, W/2
        x7 = x[:, :, 1::2, 1::2, 1::2]  # B, C, D/2, H/2, W/2
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)  # B, 8*C, D/2, H/2, W/2

        # Step 4: Channel Reduction
        x = self.reduction(x) # B, 2*C (out_channels), D/2, H/2, W/2
        return x
class SS_Conv_SSM_3D(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer= partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size = 12,
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim//2)
    self.self_attention = SS3D(d_model=hidden_dim//2, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  **kwargs)
    self.drop_path = DropPath(drop_path)
    self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm3d(hidden_dim // 2),
            nn.Conv3d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
  def forward(self, input: torch.Tensor):
        # 輸入形狀: [batch_size, channels, depth, height, width]
        input_left, input_right = input.chunk(2, dim=1)
        input_right = input_right.permute(0, 2, 3, 4, 1).contiguous()
        input_right = self.drop_path(self.self_attention(self.ln_1(input_right)))
        input_right = input_right.permute(0, 4, 1, 2, 3).contiguous()  # [batch_size, channels, depth, height, width]
        input_left = self.conv33conv33conv11(input_left)
        output = torch.cat((input_left, input_right), dim=1)
        output = channel_shuffle_3d(output, groups=2)
        return output + input

from monai.networks.blocks import SubpixelUpsample # 需要安裝 MONAI
class upconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3, upsample = True, preact = True):
        super().__init__()


        scale_factor = 2
        spatial_dims = 3
        group_norm_groups = 8
        mode = 'pixelshuffle'
        upsample_op = None

        if mode == 'interpolation':
            upsample_op = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False),
                nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True),
            )
        elif mode == 'transpose':
            padding = kernel_size // 2
            output_padding = 1 # 對於 stride=2 常見設置
            upsample_op = nn.ConvTranspose3d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=padding,
                output_padding=output_padding,
                bias=True
            )
        elif mode == 'pixelshuffle':
            conv_out_channels = ch_out * (scale_factor ** spatial_dims)
            pre_conv = nn.Conv3d(
                in_channels=ch_in,
                out_channels=conv_out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True
            )
            shuffle = SubpixelUpsample(
                spatial_dims=spatial_dims,
                in_channels=conv_out_channels,
                out_channels=ch_out,
                scale_factor=scale_factor
            )
            upsample_op = nn.Sequential(pre_conv, 
                nn.GroupNorm(group_norm_groups, conv_out_channels),
                nn.ReLU(inplace=True),
                shuffle
            )
        else:
            raise ValueError(f"Unknown upsampling mode: {mode}. Choose from 'interpolation', 'transpose', 'pixelshuffle'.")

        if preact:
            act = nn.Sequential(
                nn.GroupNorm(group_norm_groups, ch_in),
                nn.ReLU(inplace=True)
            )
            self.up = nn.Sequential(act, upsample_op)
        else:
            act = nn.Sequential(
                nn.GroupNorm(group_norm_groups, ch_out),
                nn.ReLU(inplace=True)
            )
            self.up = nn.Sequential(upsample_op, act)

    def forward(self, x):
        x = self.up(x)
        return x

class down_block(nn.Module):
    def __init__(self, ch_in, ch_out, maxpool=False, PatchMerging= True, id=True, preact=True, kernel_size=3, size = None):
        super().__init__()
        
        self.preact = preact
        self.act = nn.Sequential(nn.GroupNorm(8, ch_out), nn.ReLU(inplace=True))
        
        #-----------------------
        # downsample approach  
        #-----------------------
        if maxpool:
            downsample = nn.MaxPool3d(kernel_size=2, stride=2)
            resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        elif PatchMerging:
            downsample = PatchMerging3D(in_channels=ch_in, out_channels=ch_out)
            resblock = ResConvBlock3D(ch_in=ch_out, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        else:
            self.down = nn.Conv3d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size//2 - (1 - kernel_size % 2),
            )
            
        #-----------------------
        # encode block using SS_Conv_SSM_3D
        #-----------------------
        self.encode_block = SS_Conv_SSM_3D(
            hidden_dim=ch_out,
            drop_path=0.3,  # 可以根據需要調整
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.0,  # 可以根據需要調整
            d_state=16,  # 可以根據需要調整
            expansion_factor=1,
            mlp_drop_rate=0.0,  # 可以根據需要調整
            orientation=0,  # 可以根據需要調整
            scan_type='scan',  # 可以根據需要調整
            size=size  # 根據你的輸入尺寸調整
        )

    def forward(self, x):
        x = self.down(x)
        x = self.act(x)
        x = self.encode_block(x)
        return x

class MedSegMamba(nn.Module):
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8, # 2^(number of downsample/upsample stages in the encoder/decoder)
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256], #last element is the size which the bottleneck processes
        mamba_d_state = 64, # for vss block
        num_layers = 9,
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        drop_path_rate=0.3,
        ssm_expansion_factor=1,
        scan_type = 'scan',
        id = False, #whether to have conv1x1 in the residual path when input and output channel sizes of residual block is the same
        preact = False, #whether to have preactivation residual blocks (norm>act>conv vs conv>norm>act)
        maxpool = False, #maxpool vs strided conv
        PatchMerging = True,
        upsample = True, #upsample vs transposed conv
        full_final_block = True, # found to do better when the last resblock is large on TABSurfer, but a lot more memory
        ):
        super().__init__()
        self.hidden_dim = int((img_dim // patch_dim) ** 3)
        
        self.preconv = nn.Conv3d(img_ch, channel_sizes[1], kernel_size=3,stride=1,padding=1) if preact else (nn.Identity())
        if preact:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[1],ch_out=channel_sizes[1], id=id, preact=preact) # 96
        else:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[0],ch_out=channel_sizes[1], id=id, preact=preact) # 96
            
        self.downconv1 = down_block(channel_sizes[1], channel_sizes[2], maxpool, PatchMerging, id, preact, size = 48)

        self.downconv2 = down_block(channel_sizes[2], channel_sizes[3], maxpool, PatchMerging, id, preact, size = 24)

        self.downconv3 = down_block(channel_sizes[3], channel_sizes[4], maxpool, PatchMerging, id, preact, size = 12)

        self.gn = nn.GroupNorm(8, channel_sizes[4])
        self.relu = nn.ReLU(inplace=True)



        self.Up4 = upconv(ch_in=channel_sizes[4],ch_out=channel_sizes[3], upsample=upsample, preact=preact)
        self.Up_conv4 = ResConvBlock3D(ch_in=channel_sizes[3]*2, ch_out=channel_sizes[3], id=id, preact=preact)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        input = self.preconv(input)
        
        x1 = self.Conv1(input)

        x2 = self.downconv1(x1)

        x3 = self.downconv2(x2)

        x = self.downconv3(x3)

        #decoding
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        return d4
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

        self.feature_extractor = MedSegMamba(img_dim = 96,
                                    patch_dim = 8,
                                    img_ch = 1,
                                    output_ch = 32,
                                    channel_sizes = [1,32,64,128,256], 
                                    mamba_d_state = 64, # for vss block
                                    num_layers = 6, #6 or 9
                                    vss_version = 'v5', # None for vanilla
                                    mlp_dropout_rate = 0, #0.1
                                    attn_dropout_rate = 0, #0.1
                                    drop_path_rate=0, #0
                                    ssm_expansion_factor=1,
                                    id = False, #TABSurfer default
                                    preact = False, #TABSurfer default
                                    maxpool = False, #TABSurfer default
                                    upsample = True, #TABSurfer default
                                    full_final_block=True,
                                    scan_type='scan'
                                    ).to(device)
                     
        self.head =  ClsRegHead(in_channels=128, feature_size=128, conv_num=3, norm_type="batchnorm", act_type="ReLU")
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
if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    device = torch.device("cuda:0")
    net  = MedSegMamba(img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256], 
        mamba_d_state = 64, # for vss block
        num_layers = 6, #6 or 9
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0, #0.1
        attn_dropout_rate = 0, #0.1
        drop_path_rate=0, #0
        ssm_expansion_factor=1,
        id = False, #TABSurfer default
        preact = False, #TABSurfer default
        maxpool = False, #TABSurfer default
        upsample = True, #TABSurfer default
        full_final_block=True,
        scan_type='scan'
        ).to(device)
    
    macs, params = get_model_complexity_info(net, (1, 96, 96, 96), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
