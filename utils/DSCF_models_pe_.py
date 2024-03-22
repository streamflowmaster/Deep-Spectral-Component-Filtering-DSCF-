#
import torch.nn as nn
import torch
import math
from typing import List

import numpy as np
import torch
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch import nn, Tensor
import torch.fx
from torch.nn.init import trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        L (int): Sequence length

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.permute(0, 1, 2, 3).contiguous().view(B, L, -1)
    return x


class WindowAttention1D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wl
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))  # 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(
            [coords_l], indexing='ij'))  # 1, Wl
        coords_flatten = torch.flatten(coords, 1)  # 1, Wl
        relative_coords = coords_flatten[:, :, None] - \
                          coords_flatten[:, None, :]  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wl, Wl, 2
        relative_coords[:, :, 0] += self.window_size - \
                                    1  # shift to start from 0
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wl, Wl) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # Wl,Wl,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wl, Wl
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock1D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention1D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, L, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = 0
        pad_r = (window_size - L % window_size) % window_size
        # 右侧补长，将序列补成window_size的整倍数，此后的Lp是padding后的序列的长度
        x = F.pad(x, (0, 0, pad_l, pad_r))
        _, Lp, _ = x.shape
        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=-shift_size, dims=(1))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # 根据偏移量计算偏移之后的x
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wl, C
        #对偏移之后的序列做加窗分段，但是实际上 就是reshape了一下，把B batch的原本的序列，换成了B*n（窗的数量换成了batch，很合理啊）
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wl, C
        #依据掩码的形式来计算 加窗序列的 attention （matrix）
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size, C))

        shifted_x = window_reverse(
            attn_windows, window_size, Lp)  # B D' H' W' C
        # 加完注意力之后的结果 重新调整成 B Lp C

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1))
        else:
            x = shifted_x
        # 再把加窗平移的影响消除掉

        if pad_r > 0:
            x = x[:, :L, :].contiguous()
        # 把padding的数据丢掉
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape

        # padding
        pad_input = (L % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, L % 2))

        x0 = x[:, 0::2, :]  # B L/2 C
        x1 = x[:, 1::2, :]  # B L/2 C
        x = torch.cat([x0, x1], -1)  # B L/2 2*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


def compute_mask(L, window_size, shift_size, device):
    Lp = int(np.ceil(L / window_size)) * window_size
    img_mask = torch.zeros((1, Lp, 1), device=device)  # 1 Lp 1
    pad_size = int(Lp - L)
    if (pad_size == 0) or (pad_size + shift_size == window_size):
        segs = (slice(-window_size), slice(-window_size, -
        shift_size), slice(-shift_size, None))
    elif pad_size + shift_size > window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-seg1), slice(-seg1, -window_size),
                slice(-window_size, -shift_size), slice(-shift_size, None))
    elif pad_size + shift_size < window_size:
        seg1 = int(window_size * 2 - L + shift_size)
        segs = (slice(-window_size), slice(-window_size, -seg1),
                slice(-seg1, -shift_size), slice(-shift_size, None))
    cnt = 0
    for d in segs:
        img_mask[:, d, :] = cnt
        cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws, 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

def cat_(x1, x2):
    if len(x1.shape) == 3:
        diffY = x2.size()[2] - x1.size()[2]
        x1 = torch.nn.functional.pad(x1, (diffY // 2, diffY - diffY // 2))
    else:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
    x = torch.cat([x2, x1], dim=1)
    return x



def conv1d_7_keep_size(in_channels,out_channels):
    return nn.Conv1d(in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=7,stride=1,padding=3)

def conv1d_5_keep_size(in_channels,out_channels):
    return nn.Conv1d(in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=5,stride=1,padding=2)

def conv1d_3_keep_size(in_channels,out_channels):
    return nn.Conv1d(in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=3,stride=1,padding=1)

def conv1d_2_down_sample(in_channels,out_channels):
    return nn.Conv1d(in_channels = in_channels,
              out_channels = out_channels,
              kernel_size = 2, stride=2)

def conv1d_4_down_sample(in_channels,out_channels):
    return nn.Conv1d(in_channels = in_channels,
              out_channels = out_channels,
              kernel_size = 4, stride=2, padding=1)

def conv1d_6_down_sample(in_channels,out_channels):
    return nn.Conv1d(in_channels = in_channels,
              out_channels = out_channels,
              kernel_size=6, stride=2, padding=2)

class ResUNetBlock1d(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, inplanes, planes, bn=False,act_layer=nn.GELU ):
        super(ResUNetBlock1d, self).__init__()
        self.bn = bn
        self.conv1 = conv1d_5_keep_size(in_channels=inplanes,out_channels=planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.act_layer = act_layer()

        self.conv2 = conv1d_5_keep_size(in_channels=planes,out_channels=planes)
        self.bn2 = nn.BatchNorm1d(planes)


        self.downsample = nn.Conv1d(in_channels=inplanes,out_channels=planes,kernel_size=1)
    def forward(self, x):

        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.act_layer(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        out = self.act_layer(out)


        residual = self.downsample(x)

        out = out + residual
        out = self.act_layer(out)

        return out

class ResUnetBasicEncoder(nn.Module):

    def __init__(self, depth,dim,):
        super(ResUnetBasicEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            ResUNetBlock1d(inplanes = dim,planes=dim,bn=False)
            for i in range(depth)])

        self.downsample = conv1d_2_down_sample(in_channels=dim,out_channels=2*dim)

    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)

        x = self.downsample(x)
        return x

class SwinBasicEncoder(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock1D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, L).
        """
        # calculate attention mask for SW-MSA
        B, C, L = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        x = rearrange(x, 'b c l -> b l c')
        # Lp = int(np.ceil(L / window_size)) * window_size
        attn_mask = compute_mask(L, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, L, -1)

        if self.downsample is not None:
            x = self.downsample(x)

        x = rearrange(x, 'b l c -> b c l')
        return x

class ConvTranspose1dDecoder(nn.Module):
    def __init__(self, inplanes, outplanes,act = nn.GELU):
        super(ConvTranspose1dDecoder, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=inplanes//2,
                                           padding=2, kernel_size=6, stride=2, bias=False)
        self.conv1 = conv1d_5_keep_size(inplanes, outplanes)
        self.act = act()
        self.conv2 = conv1d_5_keep_size(outplanes, outplanes)


    def forward(self, x1, x2):

        x1 = self.upsample(x1)

        # out = torch.cat((x1, x2), dim=1)
        out = cat_(x1, x2)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        return out


class CARAFE1dDecoder(nn.Module):
    def __init__(self, inplanes, outplanes,act = nn.GELU):
        super(CARAFE1dDecoder, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=inplanes//2,
                                           padding=2, kernel_size=6, stride=2, bias=False)
        self.conv1 = conv1d_5_keep_size(inplanes, outplanes)
        self.act = act()
        self.conv2 = conv1d_5_keep_size(outplanes, outplanes)


    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # out = torch.cat((x1, x2), dim=1)
        out = cat_(x1, x2)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        return out


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The unofficial implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


class SiT1dDecoder(nn.Module):
    def __init__(self, inplanes, outplanes,act = nn.GELU):
        super(SiT1dDecoder, self).__init__()
        self.upsample = PatchPxielShuffle(dim=inplanes)
        self.con1x1 = nn.Conv1d(in_channels=inplanes,out_channels=outplanes,kernel_size=1,stride=1)
        self.sit1 = SwinTransformerBlock1D(num_heads=16,shift_size=1,dim=outplanes)
        self.act = act()
        self.sit2 = SwinTransformerBlock1D(num_heads=16, shift_size=1, dim=outplanes)
        self.inplances = inplanes
        self.outplances = outplanes

    def forward(self, x1, x2):
        x1 = x1.permute(0,2,1)
        x1 = self.upsample(x1)  # BLC
        out = cat_(x1, x2)
        out = self.con1x1(out).permute(0,2,1)
        B,L,C = out.shape
        attn_mask = compute_mask(L, window_size=7, shift_size=1, device=out.device)
        out = self.sit1(out,attn_mask)
        out = self.act(out)
        out = self.sit2(out,attn_mask)
        out = self.act(out)
        out = out.permute(0,2,1)
        return out


class PatchPxielShuffle(nn.Module):
    """ Patch shuffle Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim//2, dim//2, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, L, C).
        """
        B, L, C = x.shape
        uB, uL,uC = B, L*2, C//2
        x = x.permute(0,2,1)   #BLC-> BCL
        x = x.contiguous().view([uB, 2, uC, L])  # BCL   -> B, (u, uC),L
        x = x.permute(0, 2, 3, 1).contiguous()   # B, u, uC,L -> B, uC,L u
        x = x.view(uB, uC, uL).permute(0,2,1) #B, uC,L u -> B, uC, uL -> B, uL, uC
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0,2,1)
        return x

class PatchEmbed1D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=32, embed_dim=128, norm_layer=None,device = 'cuda:0'):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size).to(device)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim).to(device)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, L = x.size()
        if L % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - L % self.patch_size))
        x = self.proj(x)  # B E Wl
        if self.norm is not None:
            Wl = x.size(2)
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wl)

        return x

class UnPatchEmbed1D(nn.Module):

    def __init__(self, out_chans, embed_dim, patch_size,norm_layer=None,device ='cuda:0'):
        super(UnPatchEmbed1D, self).__init__()
        self.proj = torch.nn.Conv1d(in_channels=embed_dim,
                                    out_channels=patch_size*out_chans,
                                    kernel_size=1, stride=1,).to(device)
        self.patch_size = patch_size
        self.out_chans = out_chans
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim).to(device)
        else:
            self.norm = None

    def forward(self, x):
        b,e,l = x.shape
        x = self.proj(x)
        x = x.reshape(b,self.out_chans,self.patch_size*l)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Hierarchical_1d_model(nn.Module):

    def  __init__(self,inplanes = 1, outplanes = 1,encoder_name = 'SiT',embed_dim = 128,
                 decoder_name = 'TConv',layers=[3,2,2,2],sig_len = 709,device ='cuda:0',
                  mask = True,patch_size = 16):
        super(Hierarchical_1d_model, self).__init__()
        self.inplanes = inplanes
        self.inputs = inplanes
        self.sig_len = sig_len
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.device = device
        if self.encoder_name == 'SiT':
            self.patch_embed = PatchEmbed1D(in_chans=inplanes,patch_size=16,embed_dim=embed_dim,device=device)
            self.unpatchy = UnPatchEmbed1D(out_chans=outplanes, embed_dim=embed_dim, patch_size=patch_size,
                                           device=device)

            self.inplanes = embed_dim

        self.enc1 = self._make_encoder(block_name=encoder_name, planes=32, blocks=layers[0]).to(device)
        self.enc2 = self._make_encoder(block_name=encoder_name, planes=64, blocks=layers[1]).to(device)
        self.enc3 = self._make_encoder(block_name=encoder_name, planes=128, blocks=layers[2]).to(device)
        self.enc4 = self._make_encoder(block_name=encoder_name, planes=256, blocks=layers[3]).to(device)

        self.dec3 = self._make_decoder(block_name=decoder_name, dim=512).to(device)
        self.dec2 = self._make_decoder(block_name=decoder_name, dim=256).to(device)
        self.dec1= self._make_decoder(block_name=decoder_name, dim=128).to(device)
        self.upsample = self._make_upsample(block_name=decoder_name,dim=64).to(device)
        if self.encoder_name == 'SiT':
            self.dec0 = self.conv1x1 = nn.ConvTranspose1d(32, out_channels=embed_dim, kernel_size=1, stride=1,
                                                          bias=False).to(device)
        else:
            self.dec0 = self.conv1x1 = nn.ConvTranspose1d(32, out_channels=outplanes, kernel_size=1, stride=1,
                                                          bias=False).to(device)
        in_features = self._test_feature_size(inplanes)
        self.fc1 = nn.Linear(in_features=in_features,out_features=2).to(device)
        self.mask = mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, E, L], sequence
        """
        N, E, L = x.shape  # batch, embed, length
        len_keep = int(L * (1 - mask_ratio))
        len_remove = L - len_keep
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is removed
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # batch, length_k
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.unsqueeze(1).repeat(1,E,1).bool()

    def mask_operation(self,x,mask):

        x_unmask = torch.masked_fill(x,mask,0)
        aver_power = x.mean()
        noise = torch.rand(x.shape, device=x.device)*aver_power*0.01
        x_mask = torch.masked_fill(noise,~mask,0)
        # print('mask:', mask)
        # print('x_mask', x_mask)
        # print('x_unmask',x_unmask)
        return x_unmask+x_mask

    def _test_feature_size(self,inplanes):
        self.enc = nn.Sequential(
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
        ).to(self.device)
        test_input = torch.rand((1, inplanes, self.sig_len)).to(self.device)
        if self.encoder_name == 'SiT':
            test_input = self.patch_embed(test_input)
        test_output = self.enc(test_input).reshape(-1)
        return test_output.shape[0]

    def _make_encoder(self, block_name, planes, blocks):

        if block_name == 'SiT':
            layer = SwinBasicEncoder(dim=planes,depth=blocks,num_heads=16,downsample=PatchMerging)

        elif block_name == 'ResUnet':
            layer = ResUnetBasicEncoder(dim=planes,depth=blocks)

        else: layer = None
        if self.inplanes != planes:
            downsample = nn.Conv1d(self.inplanes, planes, stride=1, kernel_size=1, bias=False)
            layers = [downsample, layer]

        else:
            layers = [layer]
        self.inplanes = planes*2

        return nn.Sequential(*layers)

    def _make_upsample(se,block_name,dim):
        if block_name in ['PatchPixelShuffle', 'PPS']:
            layer = PatchPxielShuffle(dim = dim)

        elif block_name in ['TransposeConv', 'TConv']:
            layer = nn.ConvTranspose1d(in_channels=dim, out_channels=dim//2,
                                           padding=2, kernel_size=6, stride=2, bias=False)

        else:
            layer = None
        return layer

    def _make_decoder(self,block_name,dim):


        if block_name in ['PatchPixelShuffle', 'PPS']:
            layer = SiT1dDecoder(inplanes=dim,outplanes=dim//2)

        elif block_name in ['TransposeConv', 'TConv']:
            layer = ConvTranspose1dDecoder(inplanes=dim,outplanes=dim//2)
        else: layer = None
        return layer

    def batchwise_norm(self,x):
        B,C,L = x.shape
        x_ = x.reshape(B,C*L)
        x_max = torch.max(x_,dim=1,keepdim=True)[0]
        x_max = x_max.repeat(1,C*L).reshape(B,C,L)
        x = (x)/(x_max)
        return x,x_max

    def batchwise_norm_back(self,x,x_max):
        B,C,L = x.shape
        x_max = x_max[:,0,0].repeat(1,C*L).reshape(B,C,L)
        return x*x_max

    def linear_probe(self,x):
        x, x_max = self.batchwise_norm(x)
        B,C,L = x.shape
        down1 = self.enc1(x)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3).reshape(B,-1)
        return self.fc1(down4)

    def forward(self,x):
        x, x_max = self.batchwise_norm(x)
        if self.encoder_name == 'SiT':
            x = self.patch_embed(x)  # B,1,L -> B,E,N
            if self.mask:
                mask = self.random_masking(x,mask_ratio=self.mask)
                x = self.mask_operation(x,mask)
        down1 = self.enc1(x)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3)

        up3 = self.dec3(down4,down3)
        up2 = self.dec2(up3, down2)
        up1 = self.dec1(up2, down1)
        if self.decoder_name in ['PatchPixelShuffle', 'PPS']:
            up1 = up1.permute(0,2,1)
        up0 = self.upsample(up1)
        up = self.dec0(up0)
        if self.encoder_name == 'SiT':
            up = self.unpatchy(up)
        up = self.batchwise_norm_back(up,x_max)
        return up



if __name__ == '__main__':
    # net = ResUNetBlock1d(inplanes=16,planes=32)
    # net = ResUnetBasicEncoder(depth=6,dim=16)
    # net = ResUNetBasicDecoder(inplanes=16,outplanes=8)
    # net = PatchPxielShuffle(dim=8)
    # net = SiT1dDecoder(inplanes=16,outplanes=8)
    net = Hierarchical_1d_model(inplanes=8,encoder_name='SiT',decoder_name='PPS',device='cpu')
    x = torch.rand(4,8,728)
    x1 = torch.rand(4,16,364)
    # print(net(x1.permute(0,2,1),x.permute(0,2,1)).shape)
    # print(net)
    print(net(x).shape)