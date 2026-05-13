import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


def _conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        BatchNorm2d(out_channels, momentum=0.1),
        nn.ReLU(inplace=True),
    )


def get_haar_wavelet(in_channels, pool=True):
    haar_wav_l = 1 / np.sqrt(2) * np.ones((1, 2))
    haar_wav_h = 1 / np.sqrt(2) * np.ones((1, 2))
    haar_wav_h[0, 0] = -haar_wav_h[0, 0]

    haar_wav_ll = np.transpose(haar_wav_l) * haar_wav_l
    haar_wav_lh = np.transpose(haar_wav_l) * haar_wav_h
    haar_wav_hl = np.transpose(haar_wav_h) * haar_wav_l
    haar_wav_hh = np.transpose(haar_wav_h) * haar_wav_h

    filter_ll = torch.from_numpy(haar_wav_ll).unsqueeze(0)
    filter_lh = torch.from_numpy(haar_wav_lh).unsqueeze(0)
    filter_hl = torch.from_numpy(haar_wav_hl).unsqueeze(0)
    filter_hh = torch.from_numpy(haar_wav_hh).unsqueeze(0)

    conv_op = nn.Conv2d if pool else nn.ConvTranspose2d

    ll = conv_op(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    lh = conv_op(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    hl = conv_op(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    hh = conv_op(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    ll.weight.requires_grad = False
    lh.weight.requires_grad = False
    hl.weight.requires_grad = False
    hh.weight.requires_grad = False

    ll.weight.data = filter_ll.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    lh.weight.data = filter_lh.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    hl.weight.data = filter_hl.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    hh.weight.data = filter_hh.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return ll, lh, hl, hh


class HaarWaveletPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ll, self.lh, self.hl, self.hh = get_haar_wavelet(in_channels)

    def forward(self, x):
        return self.ll(x), self.lh(x), self.hl(x), self.hh(x)


class RoIAE(nn.Module):
    def __init__(
        self,
        channels,
        reduction=4,
        eps=1e-6,
        wo_geometry=False,
        wo_statistics=False,
        wo_edge=False,
    ):
        super().__init__()
        reduced_channels = max(channels // reduction, 16)
        self.eps = eps
        self.wo_geometry = wo_geometry
        self.wo_statistics = wo_statistics
        self.wo_edge = wo_edge
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False),
            BatchNorm2d(channels * 2, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels, momentum=0.1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            BatchNorm2d(channels, momentum=0.1),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            BatchNorm2d(reduced_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.geometry_proj = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.edge_proj = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.edge_norm = BatchNorm2d(channels, momentum=0.1)
        self.stat_mlp = nn.Sequential(
            nn.Linear(reduced_channels * 2, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.activation = nn.ReLU(inplace=True)
        self.register_buffer('sobel_x', torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]))
        self.register_buffer('sobel_y', torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]))

    def _sobel_magnitude(self, x):
        channels = x.shape[1]
        weight_x = self.sobel_x.repeat(channels, 1, 1, 1)
        weight_y = self.sobel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)

    def forward(self, m_feat, n_feat):
        if m_feat.shape[-2:] != n_feat.shape[-2:]:
            n_feat = F.interpolate(n_feat, size=m_feat.shape[-2:], mode='bilinear', align_corners=False)

        fused_input = torch.cat([m_feat, n_feat], dim=1)
        fused = self.activation(self.fuse(fused_input) + self.shortcut(fused_input))

        pooled_h = fused.mean(dim=3, keepdim=True).expand_as(fused)
        pooled_w = fused.mean(dim=2, keepdim=True).expand_as(fused)
        directional_context = pooled_h + pooled_w
        latent = self.reduce(directional_context)

        geometry_gate = self.geometry_proj(latent)
        latent_mean = latent.mean(dim=(2, 3))
        latent_var = latent.var(dim=(2, 3), unbiased=False)
        stat_token = torch.cat([latent_mean, torch.log(latent_var + self.eps)], dim=1)
        statistical_gate = self.stat_mlp(stat_token).unsqueeze(-1).unsqueeze(-1)
        statistical_gate = statistical_gate.expand_as(geometry_gate)

        edge_gate = self.edge_norm(self.edge_proj(self._sobel_magnitude(latent)))
        gate_terms = []
        if not self.wo_geometry:
            gate_terms.append(self.alpha * geometry_gate)
        if not self.wo_statistics:
            gate_terms.append(self.beta * statistical_gate)
        if not self.wo_edge:
            gate_terms.append(self.gamma * edge_gate)

        if gate_terms:
            gate_map = torch.sigmoid(sum(gate_terms))
        else:
            gate_map = torch.zeros_like(fused)
        enhanced = fused * gate_map + fused

        return enhanced, gate_map


class WFDStage(nn.Module):
    def __init__(self, in_channels, dropout=0.1, wo_wavelet=False):
        super().__init__()
        self.wo_wavelet = wo_wavelet
        self.content_proj = _conv_bn_relu(in_channels, in_channels)
        self.style_proj = _conv_bn_relu(in_channels, in_channels)
        self.edge_proj = _conv_bn_relu(in_channels, in_channels)
        self.wave_pool = HaarWaveletPool(in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.low_fusion = _conv_bn_relu(in_channels * 3, in_channels)
        self.style_fusion = _conv_bn_relu(in_channels * 4, in_channels)
        self.edge_fusion = _conv_bn_relu(in_channels * 4, in_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.norm = BatchNorm2d(in_channels, momentum=0.1)
        self.branch_logits = nn.Parameter(torch.zeros(3), requires_grad=True)

    def _decompose(self, feat):
        if not self.wo_wavelet:
            return self.wave_pool(feat)
        pooled = self.avg_pool(feat)
        return pooled, pooled, pooled, pooled

    def forward(self, content_feat, style_feat, edge_feat):
        spatial_size = content_feat.shape[-2:]
        if style_feat.shape[-2:] != spatial_size:
            style_feat = F.interpolate(style_feat, size=spatial_size, mode='bilinear', align_corners=False)
        if edge_feat.shape[-2:] != spatial_size:
            edge_feat = F.interpolate(edge_feat, size=spatial_size, mode='bilinear', align_corners=False)

        content = self.content_proj(content_feat)
        style = self.style_proj(style_feat)
        edge = self.edge_proj(edge_feat)

        c_ll, c_lh, c_hl, c_hh = self._decompose(content)
        s_ll, s_lh, s_hl, s_hh = self._decompose(style)
        e_ll, e_lh, e_hl, e_hh = self._decompose(edge)

        content_low = self.low_fusion(torch.cat([c_ll, s_ll, e_ll], dim=1))
        content_high_avg = (c_lh + c_hl + c_hh) / 3.0
        style_high = self.style_fusion(torch.cat([s_lh, s_hl, s_hh, content_high_avg], dim=1))
        edge_high = self.edge_fusion(torch.cat([e_lh, e_hl, e_hh, content_high_avg], dim=1))

        content_low = F.interpolate(content_low, size=spatial_size, mode='bilinear', align_corners=False)
        style_high = F.interpolate(style_high, size=spatial_size, mode='bilinear', align_corners=False)
        edge_high = F.interpolate(edge_high, size=spatial_size, mode='bilinear', align_corners=False)

        weights = F.softmax(self.branch_logits, dim=0)
        fused = weights[0] * content_low + weights[1] * style_high + weights[2] * edge_high
        out = self.norm(self.dropout(fused) + content_feat)

        aux = {
            'content_anchor': content_low,
            'weights': weights,
            'content_low': content_low,
            'style_high': style_high,
            'edge_high': edge_high,
        }
        return out, aux


class WFDDecoder(nn.Module):
    def __init__(self, fusion_channels=256, wo_wavelet=False,
                 use_stage8=True, use_stage16=True, use_stage32=True):
        super().__init__()
        self.stage32 = WFDStage(512, wo_wavelet=wo_wavelet) if use_stage32 else None
        self.stage16 = WFDStage(256, wo_wavelet=wo_wavelet) if use_stage16 else None
        self.stage8 = WFDStage(128, wo_wavelet=wo_wavelet) if use_stage8 else None
        self.transition_32_to_16 = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(fusion_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.transition_16_to_8 = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(fusion_channels // 2, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.last_aux = {}

    def forward(self, features):
        assert len(features) == 3, 'WFDDecoder 需要 [feat32, feat16, feat8] 三个尺度特征'
        feat32, feat16, feat8 = features

        if self.stage32 is not None:
            out32, aux32 = self.stage32(feat32, feat32, feat32)
        else:
            out32, aux32 = feat32, {}
        fused16 = feat16 + F.interpolate(
            self.transition_32_to_16(out32), size=feat16.shape[-2:], mode='bilinear', align_corners=False
        )
        if self.stage16 is not None:
            out16, aux16 = self.stage16(fused16, fused16, feat16)
        else:
            out16, aux16 = fused16, {}

        fused8 = feat8 + F.interpolate(
            self.transition_16_to_8(out16), size=feat8.shape[-2:], mode='bilinear', align_corners=False
        )
        if self.stage8 is not None:
            out8, aux8 = self.stage8(fused8, fused8, feat8)
            decoder_anchor = aux8['content_anchor']
        else:
            out8, aux8 = fused8, {'content_anchor': fused8}
            decoder_anchor = fused8

        self.last_aux = {
            'stage32': aux32,
            'stage16': aux16,
            'stage8': aux8,
        }
        return out8, decoder_anchor
