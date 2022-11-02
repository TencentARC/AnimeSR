import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY


class RightAlignMSConvResidualBlocks(nn.Module):
    """right align multi-scale ConvResidualBlocks, currently only support 3 scales (1, 2, 4)"""

    def __init__(self, num_in_ch=3, num_state_ch=64, num_out_ch=64, num_block=(5, 3, 2)):
        super().__init__()

        assert len(num_block) == 3
        assert num_block[0] >= num_block[1] >= num_block[2]
        self.num_block = num_block

        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_state_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.body_s1_first = nn.ModuleList()
        for _ in range(num_block[0]):
            self.body_s1_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s2_first = nn.ModuleList()
        for _ in range(num_block[1]):
            self.body_s2_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s4_first = nn.ModuleList()
        for _ in range(num_block[2]):
            self.body_s4_first.append(ResidualBlockNoBN(num_feat=num_state_ch))

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_state_ch, 2 * num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * num_out_ch, num_out_ch, 3, 1, 1, bias=True),
        )

    def up(self, x, scale=2):
        if isinstance(x, int):
            return x
        elif scale == 2:
            return self.upsample_x2(x)
        else:
            return self.upsample_x4(x)

    def forward(self, x):
        x_s1 = self.conv_s1_first(x)
        x_s2 = self.conv_s2_first(x_s1)
        x_s4 = self.conv_s4_first(x_s2)

        flag_s2 = False
        flag_s4 = False
        for i in range(0, self.num_block[0]):
            x_s1 = self.body_s1_first[i](
                x_s1 + (self.up(x_s2, 2) if flag_s2 else 0) + (self.up(x_s4, 4) if flag_s4 else 0))
            if i >= self.num_block[0] - self.num_block[1]:
                x_s2 = self.body_s2_first[i - self.num_block[0] + self.num_block[1]](
                    x_s2 + (self.up(x_s4, 2) if flag_s4 else 0))
                flag_s2 = True
            if i >= self.num_block[0] - self.num_block[2]:
                x_s4 = self.body_s4_first[i - self.num_block[0] + self.num_block[2]](x_s4)
                flag_s4 = True

        x_fusion = self.fusion(torch.cat((x_s1, self.upsample_x2(x_s2), self.upsample_x4(x_s4)), dim=1))

        return x_fusion


@ARCH_REGISTRY.register()
class MSRSWVSR(nn.Module):
    """
    Multi-Scale, unidirectional Recurrent, Sliding Window (MSRSW)
    The implementation refers to paper: Efficient Video Super-Resolution through Recurrent Latent Space Propagation
    """

    def __init__(self, num_feat=64, num_block=(5, 3, 2), netscale=4):
        super(MSRSWVSR, self).__init__()
        self.num_feat = num_feat

        # 3(img channel) * 3(prev cur nxt 3 imgs) + 3(hr img channel) * netscale * netscale + num_feat
        self.recurrent_cell = RightAlignMSConvResidualBlocks(3 * 3 + 3 * netscale * netscale + num_feat, num_feat,
                                                             num_feat + 3 * netscale * netscale, num_block)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.pixel_shuffle = nn.PixelShuffle(netscale)
        self.netscale = netscale

    def cell(self, x, fb, state):
        res = x[:, 3:6]
        # pre frame, cur frame, nxt frame, pre sr frame, pre hidden state
        inp = torch.cat((x, pixel_unshuffle(fb, self.netscale), state), dim=1)
        # the out contains both state and sr frame
        out = self.recurrent_cell(inp)
        out_img = self.pixel_shuffle(out[:, :3 * self.netscale * self.netscale]) + F.interpolate(
            res, scale_factor=self.netscale, mode='bilinear', align_corners=False)
        out_state = self.lrelu(out[:, 3 * self.netscale * self.netscale:])

        return out_img, out_state

    def forward(self, x):
        b, n, c, h, w = x.size()
        # initialize previous sr frame and previous hidden state as zero tensor
        out = x.new_zeros(b, c, h * self.netscale, w * self.netscale)
        state = x.new_zeros(b, self.num_feat, h, w)
        out_l = []
        for i in range(n):
            if i == 0:
                # there is no previous frame for the 1st frame, so reuse 1st frame as previous
                out, state = self.cell(torch.cat((x[:, i], x[:, i], x[:, i + 1]), dim=1), out, state)
            elif i == n - 1:
                # there is no next frame for the last frame, so reuse last frame as next
                out, state = self.cell(torch.cat((x[:, i - 1], x[:, i], x[:, i]), dim=1), out, state)
            else:
                out, state = self.cell(torch.cat((x[:, i - 1], x[:, i], x[:, i + 1]), dim=1), out, state)
            out_l.append(out)

        return torch.stack(out_l, dim=1)
