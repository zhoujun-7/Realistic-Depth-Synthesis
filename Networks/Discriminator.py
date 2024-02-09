
import numpy as np
import torch
import torch.nn.functional as F

from .Generator import FullyConnectedLayer, Conv2dLayer

from Torch_utils import misc
from Torch_utils.ops import upfirdn2d



#----------------------------------------------------------------------------
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
    ):
        assert in_channels in [0, tmp_channels]

        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        if in_channels == 0:
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation)
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation)
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2, resample_filter=resample_filter)
        self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2, resample_filter=resample_filter)


    def forward(self, x):

        # FromRGB.
        if self.in_channels == 0:
            x = self.fromrgb(x)

        # Main layers.
        y = self.skip(x, gain=np.sqrt(0.5))
        
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        x = y.add_(x)

        return x



#----------------------------------------------------------------------------
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x



#----------------------------------------------------------------------------
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.

        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
    ):

        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1)

    def forward(self, x):

        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        assert x.dtype == dtype
        return x



#----------------------------------------------------------------------------
class Discriminator(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.

        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.

        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}

        cur_layer_idx = 0
        cur_block_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if cur_block_idx > 0 else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, img_channels=img_channels, **block_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
            cur_block_idx += 1

        self.b4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, img_channels=img_channels, **epilogue_kwargs)

    def forward(self, x, **block_kwargs):

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x = block(x, **block_kwargs)

        x = self.b4(x)
        return x

