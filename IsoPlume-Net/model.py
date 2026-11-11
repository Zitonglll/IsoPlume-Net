"""
IsoPlume-Net: multi-modal U-Net for automated fire smoke plume delineation
from Sentinel-5P CO and VIIRS auxiliary data.

Reference:
    Li, Z. and Wooster, M. J., An automatic 'IsoPlume-Net' EO-based atmospheric
    plume delineation approach to support improved fire emission estimation,
    Science of Remote Sensing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Max pooling (stride 2) followed by a DoubleConv block."""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bilinear up-sampling, padding to the skip size, concatenation and DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class ChannelAttention(nn.Module):
    """Channel attention operator A_C in Eq. (6)."""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class DualAttentionGate(nn.Module):
    """
    CO-guided dual attention, Eq. (6).

    Channel attention is applied to the auxiliary features x first.
    Spatial attention is then computed conditioned on the CO features g.
    """

    def __init__(self, F_g, F_l, F_int, reduction=16):
        super(DualAttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        self.channel_att = ChannelAttention(F_l, reduction)

    def forward(self, g, x):
        x_channel = self.channel_att(x)

        g1 = self.W_g(g)
        x1 = self.W_x(x_channel)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x_spatial = x_channel * psi

        return x_spatial


class AuxiliaryFusion(nn.Module):
    """
    Fuses the fire, true-colour RGB and false-colour RGB features
    (32 channels each, 96 in total) using depthwise separable interactions
    within each modality, convolutional aggregation and a residual shortcut.
    The input width of 96 assumes the default 32 channels per branch.
    """

    def __init__(self, fire_channels=32, rgb_channels=32, out_channels=96):
        super(AuxiliaryFusion, self).__init__()

        self.fire_interact = nn.Sequential(
            nn.Conv2d(fire_channels, fire_channels, 3, padding=1, groups=fire_channels, bias=False),
            nn.BatchNorm2d(fire_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fire_channels, fire_channels, 1, bias=False),
            nn.BatchNorm2d(fire_channels)
        )

        self.rgb1_interact = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, groups=rgb_channels, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels)
        )

        self.rgb2_interact = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels, 3, padding=1, groups=rgb_channels, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(96, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fire_feat, rgb1_feat, rgb2_feat):
        x_orig = torch.cat([fire_feat, rgb1_feat, rgb2_feat], dim=1)

        fire_inter = self.fire_interact(fire_feat) + fire_feat
        rgb1_inter = self.rgb1_interact(rgb1_feat) + rgb1_feat
        rgb2_inter = self.rgb2_interact(rgb2_feat) + rgb2_feat

        x_cat = torch.cat([fire_inter, rgb1_inter, rgb2_inter], dim=1)
        x_fused = self.fusion(x_cat)

        x_refined = self.refine(x_fused)
        x_short = self.shortcut(x_orig)

        out = self.relu(x_refined + x_short)

        return out


class MultiScaleAttentionUNet(nn.Module):
    """

    Inputs (all 256x256):
        co_input:        S5P TCCO, 1 channel
        fire_input:      VIIRS active-fire mask, 1 channel
        rgb1_input:      VIIRS true-colour RGB, 3 channels
        false_rgb_input: VIIRS false-colour RGB, 3 channels
    Returns single-channel logits.
    """

    def __init__(self,
                 n_channels_co=1,
                 n_channels_fire=1,
                 n_channels_rgb=3,
                 n_classes=1,
                 use_attention=True,
                 use_fusion=True,
                 use_fire=True,
                 use_rgb1=True,
                 use_false_rgb=True):
        super(MultiScaleAttentionUNet, self).__init__()

        self.use_attention = use_attention
        self.use_fusion = use_fusion
        self.use_fire = use_fire
        self.use_rgb1 = use_rgb1
        self.use_false_rgb = use_false_rgb

        # CO main encoder, Eqs. (2)-(4)
        self.co_inc = DoubleConv(n_channels_co, 64)     # 64 ch, 256x256
        self.co_down1 = Down(64, 96)                    # 96 ch, 128x128
        self.co_down2 = Down(96, 192)                   # 192 ch, 64x64

        # Auxiliary encoders, 32 channels per modality
        self.inc_fire = DoubleConv(n_channels_fire, 32)
        self.inc_rgb1 = DoubleConv(n_channels_rgb, 32)
        self.inc_false_rgb = DoubleConv(n_channels_rgb, 32)

        aux_channels = 0
        if use_fire:
            aux_channels += 32
        if use_rgb1:
            aux_channels += 32
        if use_false_rgb:
            aux_channels += 32

        # The Enhanced Fusion module requires all three auxiliary branches.
        if use_fusion and (use_fire and use_rgb1 and use_false_rgb):
            self.aux_fusion = AuxiliaryFusion(
                fire_channels=32,
                rgb_channels=32,
                out_channels=96
            )
        else:
            if aux_channels > 0:
                self.aux_fusion = nn.Sequential(
                    nn.Conv2d(aux_channels, 96, 1, bias=False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True)
                )

        # Auxiliary features at scales 2 and 3 are obtained by down-sampling
        self.aux_down1 = Down(96, 192)     # 192 ch, 128x128
        self.aux_down2 = Down(192, 384)    # 384 ch, 64x64

        # CO-guided dual attention at scales 1-3
        if use_attention:
            self.attention1 = DualAttentionGate(F_g=64, F_l=96, F_int=48)
            self.attention2 = DualAttentionGate(F_g=96, F_l=192, F_int=96)
            self.attention3 = DualAttentionGate(F_g=192, F_l=384, F_int=192)

        # 1x1 fusion of CO and attended auxiliary features
        self.combine1 = nn.Sequential(
            nn.Conv2d(160, 96, 1, bias=False),      # 64 + 96
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.combine2 = nn.Sequential(
            nn.Conv2d(288, 192, 1, bias=False),     # 96 + 192
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.combine3 = nn.Sequential(
            nn.Conv2d(576, 384, 1, bias=False),     # 192 + 384
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        # Unified deep encoder
        self.deep_down1 = Down(384, 384)   # 384 ch, 32x32
        self.deep_down2 = Down(384, 768)   # 768 ch, 16x16

        # Channel expansion used only by the CO-only model (M0)
        self.co_expand1 = nn.Sequential(
            nn.Conv2d(64, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.co_expand2 = nn.Sequential(
            nn.Conv2d(96, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.co_expand3 = nn.Sequential(
            nn.Conv2d(192, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        # Decoder: four bilinear up-sampling stages
        self.up1 = Up(1152, 384)  # 768 + 384
        self.up2 = Up(768, 192)   # 384 + 384
        self.up3 = Up(384, 96)    # 192 + 192
        self.up4 = Up(192, 64)    # 96 + 96

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, co_input, fire_input, rgb1_input, false_rgb_input):

        # CO-only model
        if not (self.use_fire or self.use_rgb1 or self.use_false_rgb):
            co_feat1 = self.co_inc(co_input)
            co_feat2 = self.co_down1(co_feat1)
            co_feat3 = self.co_down2(co_feat2)

            x1 = self.co_expand1(co_feat1)
            x2 = self.co_expand2(co_feat2)
            x3 = self.co_expand3(co_feat3)

            x4 = self.deep_down1(x3)
            x5 = self.deep_down2(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

            logits = self.outc(x)
            return logits

        # Multi-modal model
        # Scale 1 (256x256)
        co_feat1 = self.co_inc(co_input)

        aux_feats = []
        if self.use_fire:
            aux_feats.append(self.inc_fire(fire_input))
        if self.use_rgb1:
            aux_feats.append(self.inc_rgb1(rgb1_input))
        if self.use_false_rgb:
            aux_feats.append(self.inc_false_rgb(false_rgb_input))

        if self.use_fusion and len(aux_feats) == 3:
            aux_feat1 = self.aux_fusion(aux_feats[0], aux_feats[1], aux_feats[2])
        else:
            aux_feat1 = self.aux_fusion(torch.cat(aux_feats, dim=1))

        if self.use_attention:
            aux_feat1_weighted = self.attention1(co_feat1, aux_feat1)
        else:
            aux_feat1_weighted = aux_feat1

        x1 = self.combine1(torch.cat([co_feat1, aux_feat1_weighted], dim=1))

        # Scale 2 (128x128)
        co_feat2 = self.co_down1(co_feat1)
        aux_feat2 = self.aux_down1(aux_feat1)

        if self.use_attention:
            aux_feat2_weighted = self.attention2(co_feat2, aux_feat2)
        else:
            aux_feat2_weighted = aux_feat2

        x2 = self.combine2(torch.cat([co_feat2, aux_feat2_weighted], dim=1))

        # Scale 3 (64x64)
        co_feat3 = self.co_down2(co_feat2)
        aux_feat3 = self.aux_down2(aux_feat2)

        if self.use_attention:
            aux_feat3_weighted = self.attention3(co_feat3, aux_feat3)
        else:
            aux_feat3_weighted = aux_feat3

        x3 = self.combine3(torch.cat([co_feat3, aux_feat3_weighted], dim=1))

        # Unified deep encoder
        x4 = self.deep_down1(x3)
        x5 = self.deep_down2(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class SimpleTwoRGBUNet(nn.Module):
    """Baseline U-Net. All inputs are concatenated at the input layer."""

    def __init__(self, n_channels_co=1, n_channels_fire=1, n_channels_rgb=3, n_classes=1):
        super(SimpleTwoRGBUNet, self).__init__()

        total_channels = n_channels_co + n_channels_fire + 2 * n_channels_rgb

        self.inc = DoubleConv(total_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, co_input, fire_input, rgb1_input, false_rgb_input):
        x = torch.cat([co_input, fire_input, rgb1_input, false_rgb_input], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

IsoPlumeNet = MultiScaleAttentionUNet


if __name__ == "__main__":
    model = IsoPlumeNet()
    co = torch.randn(1, 1, 256, 256)
    fire = torch.randn(1, 1, 256, 256)
    rgb1 = torch.randn(1, 3, 256, 256)
    false_rgb = torch.randn(1, 3, 256, 256)
    out = model(co, fire, rgb1, false_rgb)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {out.shape}, parameters: {n_params:,}")