import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value, value)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    bias = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return weight, bias


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)

    slices = []
    for index in range(kernel.size(1)):
        dilated = F.conv_transpose2d(kernel[:, index:index + 1, :, :], identity_kernel, stride=dilate_rate)
        slices.append(dilated)
    return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        return x * self.proj(x)


class GRN(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * nx + 1) * x + self.beta
        return (self.gamma * nx + 1) * x


class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
        )

        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for kernel, dilate in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    f'dil_conv_k{kernel}_{dilate}',
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel,
                        stride=1,
                        padding=(dilate * (kernel - 1) + 1) // 2,
                        dilation=dilate,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__(f'dil_bn_k{kernel}_{dilate}', get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for kernel, dilate in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{kernel}_{dilate}')
            bn = self.__getattr__(f'dil_bn_k{kernel}_{dilate}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if not hasattr(self, 'origin_bn'):
            return
        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
        for kernel, dilate in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{kernel}_{dilate}')
            bn = self.__getattr__(f'dil_bn_k{kernel}_{dilate}')
            branch_k, branch_b = fuse_bn(conv, bn)
            origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, dilate)
            origin_b += branch_b
        merged_conv = get_conv2d(
            origin_k.size(0),
            origin_k.size(0),
            origin_k.size(2),
            stride=1,
            padding=origin_k.size(2) // 2,
            dilation=1,
            groups=origin_k.size(0),
            bias=True,
        )
        merged_conv.weight.data = origin_k
        merged_conv.bias.data = origin_b
        self.lk_origin = merged_conv
        self.__delattr__('origin_bn')
        for kernel, dilate in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{kernel}_{dilate}')
            self.__delattr__(f'dil_bn_k{kernel}_{dilate}')


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RCA(nn.Module):
    def __init__(self, inp, reduction=4, dw_kernel=3, mid_min=8):
        super().__init__()
        mid = max(mid_min, inp // reduction)
        self.dwconv_hw = nn.Conv2d(inp, inp, kernel_size=dw_kernel, padding=dw_kernel // 2, groups=inp, bias=False)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_reduce = nn.Conv2d(inp, mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(mid, inp, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        loc = self.dwconv_hw(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        gated = x_h + x_w
        gated = self.conv_reduce(gated)
        gated = self.bn(gated)
        gated = self.act(gated)
        gated = self.conv_expand(gated)
        gated = self.sigmoid(gated)
        return loc * gated


class RCM(nn.Module):
    def __init__(
        self,
        dim,
        token_mixer=RCA,
        norm_layer=nn.BatchNorm2d,
        mlp_layer=ConvMlp,
        mlp_ratio=2,
        act_layer=nn.GELU,
        ls_init_value=1e-6,
        dropout=0.1,
        **token_mixer_kwargs,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, **token_mixer_kwargs)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = nn.Dropout2d(dropout)
        self.ls1 = LayerScale(dim, init_value=1)

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return self.drop_path(x) + self.ls1(shortcut)


class DynamicConvBlock(nn.Module):
    def __init__(
        self,
        dim=64,
        ctx_dim=32,
        kernel_size=7,
        mlp_ratio=4,
        drop_path=0,
        norm_layer=LayerNorm2d,
        is_first=False,
        is_last=False,
        deploy=False,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)

        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            GRN(out_dim),
        )
        self.se_layer = SEModule(out_dim)
        self.rcm = RCM(out_dim, mlp_ratio=2)
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.fusion1 = nn.Sequential(
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )

    def _forward_inner(self, x, h_x, h_r):
        low_channels = x.shape[1]
        high_channels = h_x.shape[1]

        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        fused = torch.cat([x, h_x], dim=1)
        fused = self.dwconv1(fused)
        identity = fused
        fused = self.norm1(fused)
        fused = self.fusion(fused)
        fused = self.rcm(fused)
        fused = self.se_layer(fused)
        fused = identity + self.drop_path(self.mlp(self.norm2(fused)))

        if self.is_last:
            return self.fusion1(fused), None

        low_feat, high_feat = torch.split(fused, split_size_or_sections=[low_channels, high_channels], dim=1)
        return low_feat, high_feat

    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        return self._forward_inner(x, h_x, h_r)


__all__ = ['LayerNorm2d', 'RCA', 'RCM', 'DynamicConvBlock']
