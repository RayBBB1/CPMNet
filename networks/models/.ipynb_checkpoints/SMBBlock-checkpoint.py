from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

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
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
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

class SMBBlock(nn.Module):
    """
    SegMamba BLock
    一個結合 GSC、MambaLayer 和 MlpChannel 的塊。
    執行順序: GSC -> MambaLayer -> LayerNorm -> MlpChannel -> DropPath -> Residual
    """
    def __init__(self,
                 dim: int,               # 主要維度
                 mlp_ratio: float = 2.0, # MLP 擴展比例
                 # MambaLayer 參數
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_slices: int | None = None,
                 # DropPath
                 drop_path: float = 0.0,
                 # MLP 前的歸一化層
                 norm_layer = LayerNorm, # 使用您提供的 LayerNorm
                 ):
        """
        初始化 GSCSwinMambaMLPBlock。

        Args:
            dim: 輸入和輸出的通道維度。
            mlp_ratio: MLP 中間維度相對於 dim 的比例。
            d_state: MambaLayer 的狀態維度。
            d_conv: MambaLayer 的卷積寬度。
            expand: MambaLayer 的擴展因子。
            num_slices: MambaLayer 的 nslices 參數。
            drop_path: DropPath 的概率。
            norm_layer: 用於 MLP 前歸一化的類別。
        """
        super().__init__()
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)

        # # 1. GSC 模塊
        # self.gsc = GSC(in_channels=dim)

        # 2. MambaLayer 模塊
        # MambaLayer 內部包含 LayerNorm 和殘差連接
        self.mamba = MambaLayer(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_slices=num_slices # 如果您的 MambaLayer 支持
        )

        # 3. MLP 模塊前的歸一化
        # 需要一個作用於 (B, C, D, H, W) 的 LayerNorm
        # 使用您提供的 LayerNorm 並指定 data_format
        # 確保 normalized_shape 是 C (即 dim)
        self.norm_mlp = norm_layer(dim, data_format="channels_first")

        # # 4. MLP 模塊
        # self.mlp = MlpChannel(
        #     hidden_size=dim,
        #     mlp_dim=mlp_hidden_dim
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x: 輸入張量，形狀 (B, C, D, H, W)，其中 C == dim。

        Returns:
            輸出張量，形狀與輸入相同。
        """
        # 保存初始輸入以備 MLP 之後的殘差連接
        mlp_shortcut = x

        # 1. 通過 GSC (GSC 內部有自己的殘差)
        #x_gsc = self.gsc(x)

        # 2. 通過 MambaLayer (MambaLayer 內部有自己的預歸一化和殘差)
        # 輸入 x_gsc，輸出是 mamba(norm(x_gsc)) + x_gsc
        x_mamba = self.mamba(x)

        # 3. 通過 MLP (使用預歸一化和外部殘差)
        #   a. 歸一化 Mamba 層的輸出
        normed_for_mlp = self.norm_mlp(x_mamba)
        #   b. 通過 MLP
        # mlp_out = self.mlp(normed_for_mlp)
        output = x_mamba + normed_for_mlp # 將 MLP 的結果加回到 Mamba 的輸
        return output
