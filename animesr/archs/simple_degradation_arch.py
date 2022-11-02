from torch import nn as nn

from basicsr.archs.arch_util import default_init_weights, pixel_unshuffle
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SimpleDegradationArch(nn.Module):
    """simple degradation architecture which consists several conv and non-linear layer
    it learns the mapping from HR to LR
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, downscale=2):
        """
        :param num_in_ch: input is a pseudo HR image, channel is 3
        :param num_out_ch: output is an LR image, channel is also 3
        :param num_feat: we use a small network, hidden dimension is 64
        :param downscale: suppose (h, w) is the height&width of a real-world LR video.
                          Firstly, we select the best rescaling factor (usually around 0.5) for this LR video.
                          Secondly, we obtain the pseudo HR frames and resize them to (2h, 2w).
                          To learn the mapping from pseudo HR to LR, LBO contains a pixel-unshuffle layer with
                          a scale factor of 2 to perform the downsampling at the beginning.
        """
        super(SimpleDegradationArch, self).__init__()
        num_in_ch = num_in_ch * downscale * downscale
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1),
        )
        self.downscale = downscale

        default_init_weights(self.main)

    def forward(self, x):
        x = pixel_unshuffle(x, self.downscale)
        x = self.main(x)
        return x
