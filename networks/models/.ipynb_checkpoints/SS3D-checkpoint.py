import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
import ipdb
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
from hilbert import decode, encode  # pip install numpy-hilbert-curve
from pyzorder import ZOrderIndexer

class HSCANS(nn.Module):
    def __init__(self, size=12, dim=3, scan_type='scan'):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        self.dim = dim
        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
            locs_flat = indexes
        elif 'scan' == scan_type:
            if dim == 2:
                indexes = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)
            elif dim == 3:
                indexes = indexes.reshape(size, size, size)
                for i in np.arange(1, size, step=2):
                    indexes[:, i, :] = np.flip(indexes[:, i, :], axis=1)  # flipping y
                for j in np.arange(1, size, step=2):
                    indexes[j, :, :] = np.flip(indexes[j, :, :], axis=(0, 1))
                locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == scan_type:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(img.shape), img)
        return img_decode

class SS3D(nn.Module):  # 原版 Vanilla VSS block
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 8
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L),
            torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)
        ], dim=1).view(B, 4, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        ).view(B, K, -1, L)
        out2 = torch.transpose(out_y[:, 1].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous().view(B, -1, L)
        out3 = torch.transpose(out_y[:, 2].view(B, -1, D, W, H), dim0=2, dim1=4).contiguous().view(B, -1, L)
        out4 = torch.transpose(out_y[:, 3].view(B, -1, H, D, W), dim0=3, dim1=4).contiguous().view(B, -1, L)
        out5 = torch.flip(out_y[:, 4], dims=[-1]).view(B, -1, L)  # HWD 反向 -> HWD 正向
        out6 = torch.flip(torch.transpose(out_y[:, 5].view(B, -1, W, H, D), dim0=2, dim1=3).contiguous(), dims=[-1]).view(B, -1, L)  # WHD 反向 -> HWD
        out7 = torch.flip(torch.transpose(out_y[:, 6].view(B, -1, D, W, H), dim0=2, dim1=4).contiguous(), dims=[-1]).view(B, -1, L)  # DWH 反向 -> HWD
        out8 = torch.flip(torch.transpose(out_y[:, 7].view(B, -1, H, D, W), dim0=3, dim1=4).contiguous(), dims=[-1]).view(B, -1, L)  # HDW 反向 -> HWD
        return out_y[:, 0], out2, out3, out4, out5, out6, out7, out8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class SS3D_Seq(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
            for _ in range(8)
        ], dim=0))  # (K=8, N, inner)

        self.dt_projs_weight = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).weight
            for _ in range(8)
        ], dim=0))  # (K=8, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).bias
            for _ in range(8)
        ], dim=0))  # (K=8, inner)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=8*D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=8*D)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, direction: int):
        B, C, H, W, D = x.shape
        L = H * W * D

        # 根据 direction 选择对应的方向变换
        if direction == 0:  # whd 正向
            xs = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        elif direction == 1:  # whd 反向
            xs = torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), dims=[-1])
        elif direction == 2:  # hwd 正向
            xs = x.view(B, -1, L)
        elif direction == 3:  # hwd 反向
            xs = torch.flip(x.view(B, -1, L), dims=[-1])
        elif direction == 4:  # dwh 正向
            xs = torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L)
        elif direction == 5:  # dwh 反向
            xs = torch.flip(torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), dims=[-1])
        elif direction == 6:  # hdw 正向
            xs = torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)
        elif direction == 7:  # hdw 反向
            xs = torch.flip(torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L), dims=[-1])

        xs = xs.unsqueeze(1)  # (B, 1, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight[direction:direction+1])
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight[direction:direction+1])

        xs = xs.float().view(B, -1, L)  # (B, d_inner, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (B, d_inner, L)
        Bs = Bs.float().view(B, 1, -1, L)  # (B, 1, d_state, L)
        Cs = Cs.float().view(B, 1, -1, L)  # (B, 1, d_state, L)
        Ds = self.Ds.float()[direction * self.d_inner:(direction + 1) * self.d_inner]  # (d_inner)
        As = -torch.exp(self.A_logs.float()[direction * self.d_inner:(direction + 1) * self.d_inner])  # (d_inner, d_state)
        dt_projs_bias = self.dt_projs_bias.float()[direction]  # (d_inner)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, -1, L)

        # 将输出转换回 (B, C, H, W, D) 的形式
        if direction == 0:  # whd 正向
            out = torch.transpose(out_y.view(B, -1, W, H, D), dim0=2, dim1=3).contiguous()
        elif direction == 1:  # whd 反向
            out = torch.transpose(torch.flip(out_y, dims=[-1]).view(B, -1, W, H, D), dim0=2, dim1=3).contiguous()
        elif direction == 2:  # hwd 正向
            out = out_y.view(B, -1, H, W, D)
        elif direction == 3:  # hwd 反向
            out = torch.flip(out_y, dims=[-1]).view(B, -1, H, W, D)
        elif direction == 4:  # dwh 正向
            out = torch.transpose(out_y.view(B, -1, W, H, D), dim0=2, dim1=4).contiguous()
        elif direction == 5:  # dwh 反向
            out = torch.transpose(torch.flip(out_y, dims=[-1]).view(B, -1, W, H, D), dim0=2, dim1=4).contiguous()
        elif direction == 6:  # hdw 正向
            out = torch.transpose(out_y.view(B, -1, W, H, D), dim0=3, dim1=4).contiguous()
        elif direction == 7:  # hdw 反向
            out = torch.transpose(torch.flip(out_y, dims=[-1]).view(B, -1, W, H, D), dim0=3, dim1=4).contiguous()

        return out

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape
        xz = self.in_proj(x)  # (B, H, W, D, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, D, d_inner)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, d_inner, H, W, D)
        x = self.act(self.conv3d(x))  # (B, d_inner, H, W, D)

        # 按照 whd 正反、hwd 正反、dwh 正反、hdw 正反 的顺序处理
        for direction in range(8):
            x = self.forward_core(x, direction)

        # 将最终结果转换回 (B, H, W, D, d_inner)
        y = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, D, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)  # (B, H, W, D, d_inner)

        out = self.out_proj(y)  # (B, H, W, D, d_model)
        if self.dropout is not None:
            out = self.dropout(out)

        return out

class SS3D_Seq_SameWeight_alladd(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 只使用單一組參數矩陣，不再複製 8 份
        self.x_proj_weight = nn.Parameter(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
        )  # (N, inner)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_proj.weight)  # (inner, rank)
        self.dt_projs_bias = nn.Parameter(self.dt_proj.bias)  # (inner)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=False)  # (D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=False)  # (D)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1 and not merge:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1 and not merge:
            D = repeat(D, "n1 -> r n1", r=copies)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D, W, H = x.shape
        L = D * W * H
        K = 8  # 8 個方向

        # 拼接 8 個方向的 feature
        directions = [
            torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L),  # whd 正向
            torch.flip(torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L), dims=[-1]),  # whd 反向
            torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L),  # hwd 正向
            torch.flip(torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), dims=[-1]),  # hwd 反向
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),  # dwh 正向
            torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), dims=[-1]),  # dwh 反向
            x.view(B, -1, L),  # hdw 正向
            torch.flip(x.view(B, -1, L), dims=[-1]),  # hdw 反向
        ]
        xs = torch.cat(directions, dim=2)  # (B, C, 8*L)

        # 使用單一參數矩陣處理整個序列
        x_dbl = torch.einsum("b d l, c d -> b c l", xs, self.x_proj_weight)  # (B, dt_rank + 2*d_state, 8*L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts, self.dt_projs_weight)  # (B, d_inner, 8*L)

        xs = xs.float()  # (B, d_inner, 8*L)
        dts = dts.contiguous().float()  # (B, d_inner, 8*L)
        Bs = Bs.float().view(B, -1, 8*L)  # (B, d_state, 8*L)
        Cs = Cs.float().view(B, -1, 8*L)  # (B, d_state, 8*L)
        Ds = self.Ds.float()  # (d_inner)
        As = -torch.exp(self.A_logs.float())  # (d_inner, d_state)
        dt_projs_bias = self.dt_projs_bias.float()  # (d_inner)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False,
        )  # (B, d_inner, 8*L)

        # 還原輸出：平均 8 個方向的結果
        out_y = out_y.view(B, self.d_inner, K, L)  # (B, d_inner, 8, L)
        out_y = out_y.mean(dim=2)  # (B, d_inner, L)，平均每個方向的貢獻
        out = out_y.view(B, self.d_inner, D, W, H)  # (B, d_inner, D, W, H)
        return out

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, W, H, C = x.shape
        xz = self.in_proj(x)  # (B, D, W, H, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (B, D, W, H, d_inner)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, d_inner, D, W, H)
        x = self.act(self.conv3d(x))  # (B, d_inner, D, W, H)

        # 單次處理拼接後的長序列
        y = self.forward_core(x)  # (B, d_inner, D, W, H)

        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, W, H, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)  # (B, D, W, H, d_inner)
        out = self.out_proj(y)  # (B, D, W, H, d_model)
        if self.dropout is not None:
            out = self.dropout(out)
        return out     

class SS3D_Seq_SameWeight_LastDirect(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 單一組參數矩陣
        self.x_proj_weight = nn.Parameter(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
        )  # (N, inner)
        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_proj.weight)  # (inner, rank)
        self.dt_projs_bias = nn.Parameter(self.dt_proj.bias)  # (inner)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=False)  # (D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=False)  # (D)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1 and not merge:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1 and not merge:
            D = repeat(D, "n1 -> r n1", r=copies)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D, W, H = x.shape
        L = D * W * H
        K = 8  # 8 個方向

        # 拼接 8 個方向的 feature
        directions = [
            torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L),  # whd 正向
            torch.flip(torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L), dims=[-1]),  # whd 反向
            torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L),  # hwd 正向
            torch.flip(torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L), dims=[-1]),  # hwd 反向
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),  # dwh 正向
            torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L), dims=[-1]),  # dwh 反向
            x.view(B, -1, L),  # hdw 正向
            torch.flip(x.view(B, -1, L), dims=[-1]),  # hdw 反向
        ]
        xs = torch.cat(directions, dim=2)  # (B, C, 8*L)

        # 使用單一參數矩陣處理整個序列
        x_dbl = torch.einsum("b d l, c d -> b c l", xs, self.x_proj_weight)  # (B, dt_rank + 2*d_state, 8*L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts, self.dt_projs_weight)  # (B, d_inner, 8*L)

        xs = xs.float()  # (B, d_inner, 8*L)
        dts = dts.contiguous().float()  # (B, d_inner, 8*L)
        Bs = Bs.float().view(B, -1, 8*L)  # (B, d_state, 8*L)
        Cs = Cs.float().view(B, -1, 8*L)  # (B, d_state, 8*L)
        Ds = self.Ds.float()  # (d_inner)
        As = -torch.exp(self.A_logs.float())  # (d_inner, d_state)
        dt_projs_bias = self.dt_projs_bias.float()  # (d_inner)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False,
        )  # (B, d_inner, 8*L)

        # 只取最後一個 direction 的輸出（hdw 反向）
        out_y = out_y[:, :, -L:]  # (B, d_inner, L)，取最後 L 個像素
        out = out_y.view(B, self.d_inner, D, W, H)  # (B, d_inner, D, W, H)
        return out

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, W, H, C = x.shape
        xz = self.in_proj(x)  # (B, D, W, H, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (B, D, W, H, d_inner)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, d_inner, D, W, H)
        x = self.act(self.conv3d(x))  # (B, d_inner, D, W, H)

        # 單次處理拼接後的長序列
        y = self.forward_core(x)  # (B, d_inner, D, W, H)

        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, W, H, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)  # (B, D, W, H, d_inner)
        out = self.out_proj(y)  # (B, D, W, H, d_model)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class SS3D_directionGroup(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 4 組參數，對應 4 個方向 (whd, hwd, dwh, hdw)
        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))  # (4, N, inner)

        self.dt_projs_weight = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).weight
            for _ in range(4)
        ], dim=0))  # (4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs).bias
            for _ in range(4)
        ], dim=0))  # (4, inner)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (4*D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (4*D)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=4, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=4, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, direction_idx: int):
        B, C, D, W, H = x.shape
        L = D * W * H

        # 根據 direction_idx 選擇方向並拼接正向+逆向
        if direction_idx == 0:  # whd
            xs_forward = torch.transpose(x, dim0=3, dim1=4).contiguous().view(B, -1, L)
            xs_reverse = torch.flip(xs_forward, dims=[-1])
        elif direction_idx == 1:  # hwd
            xs_forward = torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L)
            xs_reverse = torch.flip(xs_forward, dims=[-1])
        elif direction_idx == 2:  # dwh
            xs_forward = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
            xs_reverse = torch.flip(xs_forward, dims=[-1])
        elif direction_idx == 3:  # hdw
            xs_forward = x.view(B, -1, L)
            xs_reverse = torch.flip(xs_forward, dims=[-1])

        xs = torch.cat([xs_forward, xs_reverse], dim=2)  # (B, C, 2*L)

        # 使用對應的參數組
        x_dbl = torch.einsum("b d l, c d -> b c l", xs, self.x_proj_weight[direction_idx])  # (B, dt_rank + 2*d_state, 2*L)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts, self.dt_projs_weight[direction_idx])  # (B, d_inner, 2*L)

        xs = xs.float()  # (B, d_inner, 2*L)
        dts = dts.contiguous().float()  # (B, d_inner, 2*L)
        Bs = Bs.float().view(B, -1, 2*L)  # (B, d_state, 2*L)
        Cs = Cs.float().view(B, -1, 2*L)  # (B, d_state, 2*L)
        Ds = self.Ds.float()[direction_idx * self.d_inner:(direction_idx + 1) * self.d_inner]  # (d_inner)
        As = -torch.exp(self.A_logs.float()[direction_idx * self.d_inner:(direction_idx + 1) * self.d_inner])  # (d_inner, d_state)
        dt_projs_bias = self.dt_projs_bias.float()[direction_idx]  # (d_inner)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False,
        )  # (B, d_inner, 2*L)

        # 只取逆向部分的輸出
        out_y = out_y[:, :, L:]  # (B, d_inner, L)

        # 還原形狀
        if direction_idx == 0:  # whd
            out = torch.transpose(out_y.view(B, -1, D, H, W), dim0=3, dim1=4).contiguous()
        elif direction_idx == 1:  # hwd
            out = torch.transpose(out_y.view(B, -1, H, W, D), dim0=2, dim1=4).contiguous()
        elif direction_idx == 2:  # dwh
            out = torch.transpose(out_y.view(B, -1, W, D, H), dim0=3, dim1=4).contiguous()
        elif direction_idx == 3:  # hdw
            out = out_y.view(B, -1, D, W, H)

        return out

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, W, H, C = x.shape
        xz = self.in_proj(x)  # (B, D, W, H, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (B, D, W, H, d_inner)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, d_inner, D, W, H)
        x = self.act(self.conv3d(x))  # (B, d_inner, D, W, H)

        # 處理 4 個方向並相加
        outputs = []
        for direction_idx in range(4):
            out = self.forward_core(x, direction_idx)  # (B, d_inner, D, W, H)
            outputs.append(out)
        y = sum(outputs)  # (B, d_inner, D, W, H)，將 4 個逆向輸出相加

        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, W, H, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)  # (B, D, W, H, d_inner)
        out = self.out_proj(y)  # (B, D, W, H, d_model)
        if self.dropout is not None:
            out = self.dropout(out)
        return out        
class SS3D_v5(nn.Module):  # VMamba 改良版，無乘法路徑
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size=12,
            scan_type='scan',
            num_direction=8,
            orientation=0,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        if einsum:
            self.x_proj = tuple(
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                for _ in range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
            del self.x_proj
        else:
            raise Exception('have to use einsum for now lol')

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scans = HSCANS(size=size, scan_type=scan_type)
        self.num_direction = num_direction

        # 設置 transp 和 transp2 函數
        self.set_transp_functions()

    def set_transp_functions(self):
        """根據 orientation 設置 transp 和 transp2 函數"""
        if (self.orientation % 3) == 0:
            self.transp = self._transp_identity
        elif (self.orientation % 3) == 1:
            self.transp = self._transp_xy
        elif (self.orientation % 3) == 2:
            self.transp = self._transp_xz

        if (self.orientation % 6) <= 2:
            self.transp2 = self._transp_identity
        else:  # 3, 4, 5
            self.transp2 = self._transp_yz

    def _transp_identity(self, x):
        return x

    def _transp_xy(self, x):
        return torch.transpose(x, dim0=2, dim1=3)

    def _transp_xz(self, x):
        return torch.transpose(x, dim0=2, dim1=4)

    def _transp_yz(self, x):
        return torch.transpose(x, dim0=3, dim1=4)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []
        xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3, 4)), k=1, dims=(2, 4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2, 4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=-1, dims=(2, 4)), k=1, dims=(2, 3)))).contiguous().view(B, -1, L)))
        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []
        ys.append(self.transp(self.transp2(self.scans.decode(out_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))
        ys.append(self.transp(self.transp2(self.scans.decode(inv_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2, 4)), k=-1, dims=(3, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2, 4)), k=-1, dims=(3, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2, 3)), k=1, dims=(2, 4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2, 3)), k=1, dims=(2, 4)).contiguous().view(B, -1, L))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape
        x = self.in_proj(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)
        y = self.out_norm(y)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS3D_v6(nn.Module):  # VMamba 改良版，無乘法路徑
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size=12,
            scan_type='scan',
            num_direction=6,
            orientation=0,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        if einsum:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
                for _ in range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
            del self.x_proj
        else:
            raise Exception('have to use einsum for now lol')

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(num_direction)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.num_direction = num_direction

        # 設置 rot 和 unrot 函數
        self.set_rot_functions()

    def set_rot_functions(self):
        """根據 orientation 設置 rot 和 unrot 函數"""
        if (self.orientation % 8) == 0:
            self.rot = self._rot_identity
            self.unrot = self._rot_identity
        elif (self.orientation % 8) == 1:
            self.rot = self._rot_xy_90
            self.unrot = self._unrot_xy_90
        elif (self.orientation % 8) == 2:
            self.rot = self._rot_yz_90
            self.unrot = self._unrot_yz_90
        elif (self.orientation % 8) == 3:
            self.rot = self._rot_xz_minus_90
            self.unrot = self._unrot_xz_minus_90
        elif (self.orientation % 8) == 4:
            self.rot = self._rot_complex_1
            self.unrot = self._unrot_complex_1
        elif (self.orientation % 8) == 5:
            self.rot = self._rot_xz_180
            self.unrot = self._unrot_xz_180
        elif (self.orientation % 8) == 6:
            self.rot = self._rot_complex_2
            self.unrot = self._unrot_complex_2
        elif (self.orientation % 8) == 7:
            self.rot = self._rot_complex_3
            self.unrot = self._unrot_complex_3

    def _rot_identity(self, x):
        return x

    def _rot_xy_90(self, x):
        return torch.rot90(x, 1, (2, 3))

    def _unrot_xy_90(self, x):
        return torch.rot90(x, -1, (2, 3))

    def _rot_yz_90(self, x):
        return torch.rot90(x, 1, (3, 4))

    def _unrot_yz_90(self, x):
        return torch.rot90(x, -1, (3, 4))

    def _rot_xz_minus_90(self, x):
        return torch.rot90(x, -1, (2, 4))

    def _unrot_xz_minus_90(self, x):
        return torch.rot90(x, 1, (2, 4))

    def _rot_complex_1(self, x):
        return torch.transpose(torch.transpose(torch.rot90(torch.rot90(x, 2, (2, 3)), 1, (2, 4)), 2, 4), 2, 3)

    def _unrot_complex_1(self, x):
        return torch.rot90(torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), -1, (2, 4)), 2, (2, 3))

    def _rot_xz_180(self, x):
        return torch.rot90(x, 2, (2, 4))

    def _unrot_xz_180(self, x):
        return torch.rot90(x, 2, (2, 4))

    def _rot_complex_2(self, x):
        return torch.transpose(torch.transpose(torch.rot90(x, 2, (2, 3)), 3, 4), 2, 3)

    def _unrot_complex_2(self, x):
        return torch.rot90(torch.transpose(torch.transpose(x, 2, 3), 3, 4), 2, (2, 3))

    def _rot_complex_3(self, x):
        return torch.transpose(torch.transpose(torch.rot90(x, 2, (3, 4)), 2, 3), 3, 4)

    def _unrot_complex_3(self, x):
        return torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), 2, (3, 4))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []
        xs.append(self.rot(x).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 2, 4).contiguous().view(B, -1, L))
        xs.append(torch.transpose(self.rot(x), 3, 4).contiguous().view(B, -1, L))
        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []
        ys.append(self.unrot(out_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))
        ys.append(self.unrot(inv_y[:, 0].view(B, -1, H, W, D)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(out_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 1].view(B, -1, H, W, D), 2, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(out_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))
        ys.append(self.unrot(torch.transpose(inv_y[:, 2].view(B, -1, H, W, D), 3, 4)).contiguous().view(B, -1, L))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape
        x = self.in_proj(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)
        y = self.out_norm(y)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out