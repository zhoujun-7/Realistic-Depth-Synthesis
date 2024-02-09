import numpy as np
import torch
from Torch_utils import misc
from Torch_utils import persistence
from Torch_utils.ops import conv2d_resample
from Torch_utils.ops import upfirdn2d
from Torch_utils.ops import bias_act
from Torch_utils.ops import fma


#----------------------------------------------------------------------------
@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


#----------------------------------------------------------------------------
@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


#----------------------------------------------------------------------------
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


#----------------------------------------------------------------------------
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


#----------------------------------------------------------------------------
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.

        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta


        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer) # setattr(object, name, value) 将object的name属性的值设置为value，可以添加新属性

        if  num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None

        z = z*0   ######### zj
        
        if self.z_dim > 0:
            misc.assert_shape(z, [None, self.z_dim])
            x = normalize_2nd_moment(z.to(torch.float32))

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        # x = x * 0 ######### zj
        # x = None ######### zj
        return x



#----------------------------------------------------------------------------
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1):
        super().__init__()

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=False):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype))
        return x



#----------------------------------------------------------------------------
class StyleConvLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Out resolution of this layer.

        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        use_noise       = False,        # Enable noise input

        bias            = True,
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        assert not (up!=1 and down!=1), 'Confusing factor for Upsample and Downsample.' 
        self.up = up
        self.down = down

        self.resolution = resolution
        if self.up != 1 and self.down == 1:
            self.in_resolution = resolution // self.up
        elif self.up == 1 and self.down != 1:
            self.in_resolution = resolution * self.down
        else:
            self.in_resolution = self.resolution

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))    
        self.flip_weight = (self.up == 1) and (self.down == 1) # slightly faster, if True, it's Correlation, else Convolution
        self.use_bias = bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation

        self.use_noise = use_noise
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution])) 
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))


    def forward(self, x, w, noise_mode='random', gain=1):

        assert noise_mode in ['random', 'const', 'none']
        misc.assert_shape(x, [None, self.weight.shape[1], self.in_resolution, self.in_resolution])

        styles = self.affine(w)
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, down=self.down,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=self.flip_weight, fused_modconv=False)

        if self.use_bias:
            act_gain = self.act_gain * gain
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain)

        return x



""" ================================================================= """
""" ----------- Unet & Fullsize Style Synthesis Generator ----------- """

class DownsampleBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                # Number of input channels
        out_channels,               # Number of output channels
        resolution,                 # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.in_resolution = self.resolution * 2

        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, down=2, **layer_kwargs)

    def forward(self, x, **layer_kwargs):
        misc.assert_shape(x, [None, self.in_channels, self.in_resolution, self.in_resolution])

        x = self.conv0(x, **layer_kwargs)
        x = self.conv1(x, **layer_kwargs)

        return x




class StyleDownsampleBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        out_channels,                   # Number of output channels
        w_dim,                          # Intermediate latent (W) dimensionality
        resolution,                     # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.in_resolution = self.resolution * 2

        self.conv = StyleConvLayer(in_channels, out_channels, w_dim, self.in_resolution, 
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.downsample = StyleConvLayer(out_channels, out_channels, w_dim, self.resolution, down=2,
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        
        self.num_ws = 2

    def forward(self, x, ws, **layer_kwargs):

        misc.assert_shape(x, [None, self.in_channels, self.in_resolution, self.in_resolution])
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        
        w_iter = iter(ws.unbind(dim=1))
        x = self.conv(x, next(w_iter), **layer_kwargs)
        x = self.downsample(x , next(w_iter), **layer_kwargs)

        return x
    


class ConvsBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                # Number of input channels
        out_channels,               # Number of output channels
        resolution,                 # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution

        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)

    def forward(self, x, **layer_kwargs):

        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])

        x = self.conv0(x, **layer_kwargs)
        x = self.conv1(x, **layer_kwargs)

        return x



class StyleConvsBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        out_channels,                   # Number of output channels
        w_dim,                          # Intermediate latent (W) dimensionality
        resolution,                     # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution

        self.conv0 = StyleConvLayer(in_channels, out_channels, w_dim, self.resolution, 
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.conv1 = StyleConvLayer(out_channels, out_channels, w_dim, self.resolution,
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        
        self.num_ws = 2

    def forward(self, x, ws, **layer_kwargs):

        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        
        w_iter = iter(ws.unbind(dim=1))
        x = self.conv0(x, next(w_iter), **layer_kwargs)
        x = self.conv1(x , next(w_iter), **layer_kwargs)

        return x



class UpsampleBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                # Number of input channels
        out_channels,               # Number of output channels
        resolution,                 # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.in_resolution = self.resolution // 2

        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation, up=2, **layer_kwargs)

    def forward(self, x, **layer_kwargs):
        misc.assert_shape(x, [None, self.in_channels, self.in_resolution, self.in_resolution])

        x = self.conv0(x, **layer_kwargs)
        x = self.conv1(x, **layer_kwargs)

        return x




class StyleUpsampleBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels
        out_channels,                   # Number of output channels
        w_dim,                          # Intermediate latent (W) dimensionality
        resolution,                     # Out resolution of this block

        kernel_size     = 3,
        bias            = True,
        activation      = 'lrelu',
        **layer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.in_resolution = self.resolution // 2

        self.conv = StyleConvLayer(in_channels, out_channels, w_dim, self.in_resolution, 
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        self.upsample = StyleConvLayer(out_channels, out_channels, w_dim, self.resolution, up=2,
                        kernel_size=kernel_size, bias=bias, activation=activation, **layer_kwargs)
        
        self.num_ws = 2

    def forward(self, x, ws, **layer_kwargs):

        misc.assert_shape(x, [None, self.in_channels, self.in_resolution, self.in_resolution])
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        
        w_iter = iter(ws.unbind(dim=1))
        x = self.conv(x, next(w_iter), **layer_kwargs)
        x = self.upsample(x , next(w_iter), **layer_kwargs)

        return x



class StyleSynNet(torch.nn.Module):
    def __init__(self,
        w_dim,
        img_resolution,
        img_channels,

        channel_base        = 16384,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.

        unet_scales         = 2,        # The lowest scale for unet, lowest res = img_res // (2**scale)
        fullsize_modules    = 2,        # Number of fullsize generation modules
        unetDownsample      = 'DownsampleBlock',
        unetConvs           = 'ConvsBlock',
        unetUpsample        = 'StyleUpsampleBlock',
        fullsizeConvs       = 'StyleConvsBlock',
        **block_kwargs
    ):
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.unet_scales = unet_scales
        self.num_ws = 0

        unetDownsample_kwargs = dict(w_dim=w_dim) if 'Style' in unetDownsample else dict()
        unetDownsample = {'DownsampleBlock':DownsampleBlock, 'StyleDownsampleBlock':StyleDownsampleBlock}[unetDownsample]
        unetConvs_kwargs = dict(w_dim=w_dim) if 'Style' in unetConvs else dict()
        unetConvs = {'ConvsBlock':ConvsBlock, 'StyleConvsBlock':StyleConvsBlock}[unetConvs]
        unetUpsample_kwargs = dict(w_dim=w_dim) if 'Style' in unetUpsample else dict()
        unetUpsample = {'UpsampleBlock':UpsampleBlock, 'StyleUpsampleBlock':StyleUpsampleBlock}[unetUpsample]
        fullsizeConvs_kwargs = dict(w_dim=w_dim) if 'Style' in fullsizeConvs else dict()
        fullsizeConvs = {'ConvsBlock':ConvsBlock, 'StyleConvsBlock':StyleConvsBlock}[fullsizeConvs]

        main_channels = channel_base // img_resolution
        unet_res_list = [img_resolution // (2**scale) for scale in range(unet_scales+1)]
        unet_channels_dict = {res:min(channel_max, channel_base//res) for res in unet_res_list}


        self.fromRgb = Conv2dLayer(img_channels, main_channels, kernel_size=1, activation='lrelu', bias=True)

        self.unet_modules = []
        # unet downsample
        for ri, ro in zip(unet_res_list[:-1], unet_res_list[1:]):
            ci, co = unet_channels_dict[ri], unet_channels_dict[ro]
            unetDownsample_kwargs.update(dict(in_channels=ci, out_channels=co, resolution=ro))
            block = unetDownsample(**unetDownsample_kwargs, **block_kwargs)
            setattr(self, f'res{ri}To{ro}', block)
            self.unet_modules.append(f'res{ri}To{ro}')

        # unet convs
        ri = ro = unet_res_list[-1]
        ci = co = unet_channels_dict[ri]
        unetConvs_kwargs.update(dict(in_channels=ci, out_channels=co, resolution=ro))
        block = unetConvs(**unetConvs_kwargs , **block_kwargs)
        setattr(self, f'res{ri}To{ro}', block)
        self.unet_modules.append(f'res{ri}To{ro}')

        # unet upsample
        for ri, ro in zip(unet_res_list[::-1][:-1], unet_res_list[::-1][1:]):
            ci, co = unet_channels_dict[ri], unet_channels_dict[ro]
            unetUpsample_kwargs.update(dict(in_channels=ci*2, out_channels=co, resolution=ro))
            block = unetUpsample(**unetUpsample_kwargs, **block_kwargs)
            setattr(self, f'res{ri}To{ro}', block)
            self.unet_modules.append(f'res{ri}To{ro}')

        # unet toRGB
        block = ToRGBLayer(co, img_channels, w_dim)
        setattr(self, 'unet_toRgb', block)
        self.num_ws += 1

        # other modules
        self.fullsize_modules = []
        for idx in range(1, fullsize_modules+1):
            fullsizeConvs_kwargs.update(dict(in_channels=main_channels, out_channels=main_channels, resolution=ro))
            block = fullsizeConvs(**fullsizeConvs_kwargs, **block_kwargs) # convs
            setattr(self, f'fullsizeConvs{idx}', block)
            self.fullsize_modules.append(f'fullsizeConvs{idx}')

            block = ToRGBLayer(main_channels, img_channels, w_dim)  # toRgbs
            setattr(self, f'toRgb{idx}', block)
            self.fullsize_modules.append(f'toRgb{idx}')
            self.num_ws += 1

        # num_ws
        for name in self.unet_modules + self.fullsize_modules:
            block = getattr(self, name)
            if hasattr(block, 'num_ws'):
                self.num_ws += block.num_ws


    def forward(self, img, ws):

        img = img.to(torch.float32)
        ws = ws.to(torch.float32)

        features = self.fromRgb(img)
        ws_idx = 0 
        
        # unet downsample
        temp_values = {}
        for name in self.unet_modules[:self.unet_scales]:
            block = getattr(self, name)
            if hasattr(block, 'num_ws'):
                wx = ws.narrow(1, ws_idx, block.num_ws); ws_idx += block.num_ws
                features = block(features, wx)
            else:
                features = block(features)
            temp_values[block.resolution] = features.clone()
        
        # unet convs
        name = self.unet_modules[self.unet_scales]
        block = getattr(self, name)
        if hasattr(block, 'num_ws'):
            wx = ws.narrow(1, ws_idx, block.num_ws); ws_idx += block.num_ws
            features = block(features, wx)
        else:
            features = block(features)
        
        # unet upsample
        for name in self.unet_modules[-self.unet_scales:]:
            block = getattr(self, name)
            features = torch.cat([features, temp_values[block.in_resolution]], dim=1)
            if hasattr(block, 'num_ws'):
                wx = ws.narrow(1, ws_idx, block.num_ws); ws_idx += block.num_ws
                features = block(features, wx)
            else:
                features = block(features)

        # unet toRgb
        block = getattr(self, 'unet_toRgb')
        wx = ws.narrow(1, ws_idx, 1)[:,0,:]; ws_idx += 1
        rgb = block(features, wx)

        # fullsize modules
        idx = -1
        num_modules = len(self.fullsize_modules)
        while idx < (num_modules - 1):

            idx += 1
            block = getattr(self, self.fullsize_modules[idx]) # convs
            if hasattr(block, 'num_ws'):
                wx = ws.narrow(1, ws_idx, block.num_ws); ws_idx += block.num_ws
                features = block(features, wx)
            else:
                features = block(features)
            
            idx += 1
            block = getattr(self, self.fullsize_modules[idx])
            wx = ws.narrow(1, ws_idx, 1)[:,0,:]; ws_idx += 1
            rgb = rgb.add(block(features, wx))

        return rgb



class StyleSynGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = StyleSynNet(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, nmaps, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z,truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(nmaps, ws, **synthesis_kwargs)
        return img