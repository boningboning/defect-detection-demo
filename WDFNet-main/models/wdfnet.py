# ------------------------------------------------------------------------------
# WDFNet: Wavelet Decoupling and Fusion Network
# ------------------------------------------------------------------------------
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .wdf_blocks import DynamicConvBlock, LayerNorm2d
from .wfd_decoder import RoIAE, WFDDecoder

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class WDFNet(nn.Module):
    def __init__(
        self,
        m=3,
        n=4,
        num_classes=2,
        planes=64,
        ppm_planes=112,
        head_planes=256,
        sub_depth=[1, 1],
        augment=True,
        embed_dim=[256, 512, 512, 512],
        depth=[2, 2, 2, 2],
        kernel_size=[7, 7, 7, 7],
        sub_mlp_ratio=[4, 4],
        sub_num_heads=[4, 8],
        res_scale=True,
        smk_size=5,
        deploy=False,
        use_gemm=True,
        drop_path_rate=0.5,
        norm_layer=LayerNorm2d,
        use_checkpoint=[0, 0, 0, 0],
        roiae_wo_geometry=False,
        roiae_wo_statistics=False,
        roiae_wo_edge=False,
        wfd_wo_wavelet=False,
        roiae_use_16=True,
        roiae_use_32=True,
        wfd_use_8=True,
        wfd_use_16=True,
        wfd_use_32=True,
    ):
        super().__init__()
        self.augment = augment
        self.roiae_use_16 = roiae_use_16
        self.roiae_use_32 = roiae_use_32
        self.wfd_decoder = WFDDecoder(
            fusion_channels=256,
            wo_wavelet=wfd_wo_wavelet,
            use_stage8=wfd_use_8,
            use_stage16=wfd_use_16,
            use_stage32=wfd_use_32,
        )
        self.roiae16_ctx_proj = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        ) if roiae_use_16 else None
        self.roiae16 = RoIAE(
            planes * 4,
            wo_geometry=roiae_wo_geometry,
            wo_statistics=roiae_wo_statistics,
            wo_edge=roiae_wo_edge,
        ) if roiae_use_16 else None
        self.roiae32 = RoIAE(
            planes * 8,
            wo_geometry=roiae_wo_geometry,
            wo_statistics=roiae_wo_statistics,
            wo_edge=roiae_wo_edge,
        ) if roiae_use_32 else None
        self.last_aux = {}

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer6 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[0],
                    ctx_dim=embed_dim[1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[0],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i == 0),
                    is_last=(i == sub_depth[0] - 1),
                    use_checkpoint=(i < use_checkpoint[2]),
                )
            )

        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[1],
                    res_scale=res_scale,
                    drop_path=dpr[i + sum(depth) + sub_depth[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    is_first=(i == 0),
                    is_last=(i == sub_depth[1] - 1),
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i < use_checkpoint[3]),
                )
            )

        self.pag4 = PagFM(planes * 4, planes * 4)
        self.high_level_proj = nn.Conv2d(planes * 14, planes * 2, kernel_size=1)
        self.roiae_aux_proj = nn.Conv2d(planes * 2, planes * 2, kernel_size=1)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 6, planes * 6, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(BasicBlock, planes * 2, planes * 2, 1)

        if self.augment:
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_i = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_il = segmenthead(planes * 4, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        x_d = self.layer3_d(x)
        x = self.relu(self.layer3(x))
        x_16 = x.clone()
        x = self.relu(self.layer4(x))
        x_32 = x.clone()

        ctx_up = F.interpolate(x_32, size=x_16.shape[-2:], mode='bilinear', align_corners=False)
        if self.roiae16 is not None:
            ctx_up_for_roiae = self.roiae16_ctx_proj(ctx_up)
            x_16, gate16 = self.roiae16(x_16, ctx_up_for_roiae)
        else:
            gate16 = None
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                x = ctx_up
            x_16, x = blk(x_16, x, ctx_up)

        x_d = x_d + F.interpolate(
            self.diff3(x_16), size=x_d.shape[-2:], mode='bilinear', align_corners=algc
        )
        x_d = self.layer4_d(self.relu(x_d))

        ctx2 = self.relu(self.layer5(x_16))
        if self.roiae32 is not None:
            x_32, gate32 = self.roiae32(x_32, ctx2)
        else:
            gate32 = None
        for _, blk in enumerate(self.sub_blocks4):
            x_32, ctx2 = blk(x_32, ctx2, ctx2)

        x_d = x_d + F.interpolate(
            self.diff4(x_32), size=x_d.shape[-2:], mode='bilinear', align_corners=algc
        )

        if self.augment:
            temp_d = x_d

        x_8 = self.layer5_d(self.relu(x_d))
        decoder_out, decoder_anchor = self.wfd_decoder([x_32, x_16, x_8])
        self.last_aux = {
            'gate16': gate16,
            'gate32': gate32,
            'decoder': self.wfd_decoder.last_aux,
        }

        if self.augment:
            x_extra_p = self.seghead_p(self.roiae_aux_proj(decoder_anchor))
            x_extra_d = self.seghead_d(temp_d)
            x_extra_i = self.seghead_i(decoder_out)
            return [x_extra_p, x_extra_i, x_extra_d]
        return decoder_out


def _resolve_variant(name):
    name = name.lower()
    if 'small' in name or name.endswith('_s') or name.endswith('-s'):
        return 'small'
    if 'medium' in name or name.endswith('_m') or name.endswith('-m'):
        return 'medium'
    if 'large' in name or name.endswith('_l') or name.endswith('-l'):
        return 'large'
    if name.endswith('s'):
        return 'small'
    if name.endswith('m'):
        return 'medium'
    if name.endswith('l'):
        return 'large'
    return 'large'


def _build_wdfnet(
    name,
    num_classes,
    augment,
    roiae_wo_geometry=False,
    roiae_wo_statistics=False,
    roiae_wo_edge=False,
    wfd_wo_wavelet=False,
    roiae_use_16=True,
    roiae_use_32=True,
    wfd_use_8=True,
    wfd_use_16=True,
    wfd_use_32=True,
):
    variant = _resolve_variant(name)
    common_kwargs = dict(
        num_classes=num_classes,
        augment=augment,
        roiae_wo_geometry=roiae_wo_geometry,
        roiae_wo_statistics=roiae_wo_statistics,
        roiae_wo_edge=roiae_wo_edge,
        wfd_wo_wavelet=wfd_wo_wavelet,
        roiae_use_16=roiae_use_16,
        roiae_use_32=roiae_use_32,
        wfd_use_8=wfd_use_8,
        wfd_use_16=wfd_use_16,
        wfd_use_32=wfd_use_32,
    )
    if variant == 'small':
        return WDFNet(m=3, n=3, planes=64, ppm_planes=96, head_planes=128, **common_kwargs)
    if variant == 'medium':
        return WDFNet(m=2, n=3, planes=64, ppm_planes=96, head_planes=128, **common_kwargs)
    return WDFNet(m=3, n=4, planes=64, ppm_planes=112, head_planes=256, **common_kwargs)


def get_seg_model(cfg, imgnet_pretrained):
    model = _build_wdfnet(
        cfg.MODEL.NAME,
        cfg.DATASET.NUM_CLASSES,
        augment=True,
        roiae_wo_geometry=cfg.MODEL.ROIAE_WO_GEOMETRY,
        roiae_wo_statistics=cfg.MODEL.ROIAE_WO_STATISTICS,
        roiae_wo_edge=cfg.MODEL.ROIAE_WO_EDGE,
        wfd_wo_wavelet=cfg.MODEL.WFD_WO_WAVELET,
        roiae_use_16=cfg.MODEL.ROIAE_USE_16,
        roiae_use_32=cfg.MODEL.ROIAE_USE_32,
        wfd_use_8=cfg.MODEL.WFD_USE_8,
        wfd_use_16=cfg.MODEL.WFD_USE_16,
        wfd_use_32=cfg.MODEL.WFD_USE_32,
    )
    if not cfg.MODEL.PRETRAINED:
        return model

    if not os.path.isfile(cfg.MODEL.PRETRAINED):
        logging.warning('Pretrained weights not found: %s', cfg.MODEL.PRETRAINED)
        return model

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        model_dict = model.state_dict()
        pretrained_state = {
            k: v for k, v in pretrained_state.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_state)
        logging.info('Loaded %d parameters from %s', len(pretrained_state), cfg.MODEL.PRETRAINED)
        model.load_state_dict(model_dict, strict=False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {
            k[6:]: v
            for k, v in pretrained_dict.items()
            if k.startswith('model.') and k[6:] in model_dict and v.shape == model_dict[k[6:]].shape
        }
        model_dict.update(pretrained_dict)
        logging.info('Loaded %d parameters from %s', len(pretrained_dict), cfg.MODEL.PRETRAINED)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):
    return _build_wdfnet(name, num_classes, augment=False)


if __name__ == '__main__':
    device = torch.device('cuda')
    model = get_pred_model(name='wdfnet_small', num_classes=4)
    model.eval()
    model.to(device)
    iterations = None

    input_tensor = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input_tensor)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            fps = iterations / elapsed_time
            iterations = int(fps * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input_tensor)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    fps = 1000 / latency
    print(fps)
