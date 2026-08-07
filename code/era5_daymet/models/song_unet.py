#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 本文件的网络结构源自 NVIDIA PhysicsNeMo (Apache-2.0):
#   SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
#   SPDX-License-Identifier: Apache-2.0
#   release v1.2.0, commit 17efe3e3f4d179fc59e822543697add589426147
#   physicsnemo/models/diffusion/{layers.py, song_unet.py, utils.py}
"""
============================================================================
song_unet.py — CorrDiff 骨干网络(SongUNet), 与官方实现逐值等价
============================================================================
从官方源码移植, 数学部分未作改动; 仅去掉 physicsnemo 的 Module/ModelMetaData 基类
与 nvtx 性能标注这两项与计算无关的依赖, 使其可在本仓库环境中独立运行。

`SongUNet` 是 EDM/CorrDiff 论文使用的骨干: UNetBlock 带 skip_scale=sqrt(0.5) 的残差
缩放、低通滤波重采样、逐层缩放的权重初始化, 瓶颈层无条件带自注意力。
`SongUNetPosEmbd` 在其上追加全域位置网格通道(sinusoidal / learnable)。

规模由 model_channels 与 channel_mult 控制; channel_mult 的长度即分辨率级数,
下采样次数为长度减一。
============================================================================
"""
import contextlib
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from torch.nn.functional import elu, gelu, leaky_relu, relu, sigmoid, silu, tanh
from torch.utils.checkpoint import checkpoint

_is_apex_available = False       # 本仓库不依赖 apex 的融合 GroupNorm


@contextlib.contextmanager
def _nvtx_noop(*args, **kwargs):
    """官方源码用 nvtx 做性能标注; 此处替换为空上下文, 不影响数值。"""
    yield


def weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


def _recursive_property(prop_name: str, prop_type: type, doc: str) -> property:
    """
    Property factory that sets the property on a Module ``self`` and
    recursively on all submodules.
    For ``self``, the property is stored under a semi-private ``_<prop_name>`` attribute
    and for submodules the setter is delegated to the ``setattr`` function.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    prop_type : type
        The type of the property.
    doc : str
        The documentation string for the property.

    Returns
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        if not isinstance(value, prop_type):
            raise TypeError(
                f"{prop_name} must be a {prop_type.__name__} value, but got {type(value).__name__}."
            )
        # Set for self
        setattr(self, f"_{prop_name}", value)
        # Set for submodules
        submodules = iter(self.modules())
        next(submodules)  # Skip self
        for m in submodules:
            if hasattr(m, prop_name):
                setattr(m, prop_name, value)

    def _getter(self):
        return getattr(self, f"_{prop_name}")

    return property(_getter, _setter, doc=doc)


def _validate_amp(amp_mode: bool) -> None:
    """Raise if `amp_mode` is False but PyTorch autocast (CPU or CUDA) is active.

    Parameters
    ----------
    amp_mode : bool
        Your intended AMP flag. Set False when you require full precision.
    """

    try:
        cuda_amp = bool(torch.is_autocast_enabled())
    except AttributeError:  # very old PyTorch
        cuda_amp = False
    try:
        cpu_amp = bool(torch.is_autocast_enabled("cpu"))
    except AttributeError:
        cpu_amp = False

    if not amp_mode and (cuda_amp or cpu_amp):
        active = []
        if cuda_amp:
            active.append("cuda")
        if cpu_amp:
            active.append("cpu")
        raise RuntimeError(
            f"amp_mode=False but torch autocast is enabled on: {', '.join(active)}. "
            "Disable autocast for this region or set amp_mode=True if mixed precision is intended."
        )


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.amp_mode = amp_mode
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
        x = x @ weight.t()
        if self.bias is not None:
            x = x.add_(bias)
        return x


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of conv2d. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        fused_conv_bias: bool = False,
        amp_mode: bool = False,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        if not kernel and fused_conv_bias:
            print(
                "Warning: Kernel is required when fused_conv_bias is enabled. Setting fused_conv_bias to False."
            )
            fused_conv_bias = False

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.fused_conv_bias = fused_conv_bias
        self.amp_mode = amp_mode
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        weight, bias, resample_filter = self.weight, self.bias, self.resample_filter
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
            if (
                self.resample_filter is not None
                and self.resample_filter.dtype != x.dtype
            ):
                resample_filter = self.resample_filter.to(x.dtype)

        w = weight if weight is not None else None
        b = bias if bias is not None else None
        f = resample_filter if resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x, w, padding=max(w_pad - f_pad, 0), bias=b
                )
            else:
                x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:  # ask in corrdiff channel whether w will ever be none
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.
    use_apex_gn : bool, optional
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu. Defaults to False.
    fused_act : bool, optional
        Whether to fuse the activation function with GroupNorm. Defaults to False.
    act : str, optional
        The activation function to use when fusing activation with GroupNorm. Defaults to None.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
        use_apex_gn: bool = False,
        fused_act: bool = False,
        act: str = None,
        amp_mode: bool = False,
    ):
        if fused_act and act is None:
            raise ValueError("'act' must be specified when 'fused_act' is set to True.")

        super().__init__()
        self.num_groups = min(
            num_groups,
            (num_channels + min_channels_per_group - 1) // min_channels_per_group,
        )
        if num_channels % self.num_groups != 0:
            raise ValueError(
                "num_channels must be divisible by num_groups or min_channels_per_group"
            )
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        if use_apex_gn and not _is_apex_available:
            raise ValueError("'apex' is not installed, set `use_apex_gn=False`")
        self.use_apex_gn = use_apex_gn
        self.fused_act = fused_act
        self.act = act.lower() if act else act
        self.act_fn = None
        self.amp_mode = amp_mode
        if self.use_apex_gn:
            if self.fused_act:
                self.gn = ApexGroupNorm(
                    num_groups=self.num_groups,
                    num_channels=num_channels,
                    eps=self.eps,
                    affine=True,
                    act=self.act,
                )

            else:
                self.gn = ApexGroupNorm(
                    num_groups=self.num_groups,
                    num_channels=num_channels,
                    eps=self.eps,
                    affine=True,
                )
        if self.fused_act:
            self.act_fn = self.get_activation_function()

    def forward(self, x):
        if (not x.is_cuda) and self.use_apex_gn:
            warnings.warn(
                "Apex GroupNorm is not supported on CPU. Please move your tensor to GPU or disable use_apex_gn."
            )
        weight, bias = self.weight, self.bias
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if not self.use_apex_gn:
                if weight.dtype != x.dtype:
                    weight = self.weight.to(x.dtype)
                if bias.dtype != x.dtype:
                    bias = self.bias.to(x.dtype)
        if self.use_apex_gn:
            x = self.gn(x)
        elif self.training:
            # Use default torch implementation of GroupNorm for training
            # This does not support channels last memory format
            x = torch.nn.functional.group_norm(
                x,
                num_groups=self.num_groups,
                weight=weight,
                bias=bias,
                eps=self.eps,
            )
            if self.fused_act:
                x = self.act_fn(x)
        else:
            # Use custom GroupNorm implementation that supports channels last
            # memory layout for inference
            x = rearrange(x, "b (g c) h w -> b g c h w", g=self.num_groups)

            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)

            x = (x - mean) * (var + self.eps).rsqrt()
            x = rearrange(x, "b g c h w -> b (g c) h w")

            weight = rearrange(weight, "c -> 1 c 1 1")
            bias = rearrange(bias, "c -> 1 c 1 1")
            x = x * weight + bias

            if self.fused_act:
                x = self.act_fn(x)
        return x

    def get_activation_function(self):
        """
        Get activation function given string input
        """

        activation_map = {
            "silu": silu,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "gelu": gelu,
            "elu": elu,
        }

        act_fn = activation_map.get(self.act, None)
        if act_fn is None:
            raise ValueError(f"Unknown activation function: {self.act}")
        return act_fn


class AttentionOp(torch.autograd.Function):
    """
    Attention weight computation, i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.
    """

    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / torch.sqrt(torch.tensor(k.shape[1]))).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )

        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


class Attention(torch.nn.Module):
    """
    Self-attention block used in U-Net-style architectures, such as DDPM++, NCSN++, and ADM.
    Applies GroupNorm followed by multi-head self-attention and a projection layer.

    Parameters
    ----------
    out_channels : int
        Number of channels :math:`C` in the input and output feature maps.
    num_heads : int
        Number of attention heads. Must be a positive integer.
    eps : float, optional, default=1e-5
        Epsilon value for numerical stability in GroupNorm.
    init_zero : dict, optional, default={'init_weight': 0}
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=None
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    init : dict, optional, default={}
        Initialization parameters for convolutional and linear layers.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is enabled.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.


    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C, H, W)`, where :math:`B` is batch
        size, :math:`C` is `out_channels`, and :math:`H, W` are spatial
        dimensions.

    Outputs
    -------
    torch.Tensor
        Output tensor of the same shape as input: :math:`(B, C, H, W)`.
    """

    def __init__(
        self,
        *,
        out_channels: int,
        num_heads: int,
        eps: float = 1e-5,
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        init: Dict[str, Any] = dict(),
        use_apex_gn: bool = False,
        amp_mode: bool = False,
        fused_conv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm = GroupNorm(
            num_channels=out_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            amp_mode=amp_mode,
        )
        self.qkv = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 3,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **(init_attn if init_attn is not None else init),
        )
        self.proj = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"`num_heads` must be a positive integer, but got {num_heads}"
            )
        self.num_heads = num_heads

    def forward(self, x):
        q, k, v = (
            torch.permute(
                self.qkv(self.norm(x)), (0, 2, 3, 1)
            )  # [B,H,W,C*3] in memory layout, shape [B,C*3,H,W] -> [B,H,W,C*3]
            .reshape(  # -> [X.shape[0], heads, (H*W), C//heads, 3]
                x.shape[0],
                self.num_heads,
                x.shape[2] * x.shape[3],
                x.shape[1] // self.num_heads,
                3,
            )
            .unbind(-1)
        )  # [B, heads, H*W, C//heads]

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(k.shape[-1])
        )
        attn = attn.transpose(-1, -2)  # [B, 1, C, H*W]
        x = self.proj(attn.reshape(*x.shape)).add_(x)
        return x


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional
        If True, applies upsampling in the forward pass. By default False.
    down : bool, optional
        If True, applies downsampling in the forward pass. By default False.
    attention : bool, optional
        If True, enables the self-attention mechanism in the block. By default False.
    num_heads : int, optional
        Number of attention heads. If None, defaults to `out_channels // 64`.
    channels_per_head : int, optional
        Number of channels per attention head. By default 64.
    dropout : float, optional
        Dropout probability. By default 0.0.
    skip_scale : float, optional
        Scale factor applied to skip connections. By default 1.0.
    eps : float, optional
        Epsilon value used for normalization layers. By default 1e-5.
    resample_filter : List[int], optional
        Filter for resampling layers. By default [1, 1].
    resample_proj : bool, optional
        If True, resampling projection is enabled. By default False.
    adaptive_scale : bool, optional
        If True, uses adaptive scaling in the forward pass. By default True.
    init : dict, optional
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional
        Initialization parameters with zero weights for certain layers. By default
        {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    use_apex_gn : bool, optional
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu. Defaults to False.
    act : str, optional
        The activation function to use when fusing activation with GroupNorm. Defaults to None.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of conv2d. By default False.
    profile_mode:
        A boolean flag indicating whether to enable all nvtx annotations during profiling.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    # NOTE: these attributes have specific usage in old checkpoints, do not
    # reuse them!
    _reserved_attributes: Set[str] = set(["norm2", "qkv", "proj"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        use_apex_gn: bool = False,
        act: str = "silu",
        fused_conv_bias: bool = False,
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.attention = attention
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode
        self.norm0 = GroupNorm(
            num_channels=in_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            fused_act=True,
            act=act,
            amp_mode=amp_mode,
        )
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            amp_mode=amp_mode,
            **init,
        )
        if self.adaptive_scale:
            self.norm1 = GroupNorm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
            )
        else:
            self.norm1 = GroupNorm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                act=act,
                fused_act=True,
                amp_mode=amp_mode,
            )
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            fused_conv_bias = fused_conv_bias if kernel != 0 else False
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                fused_conv_bias=fused_conv_bias,
                amp_mode=amp_mode,
                **init,
            )

        if self.attention:
            self.attn = Attention(
                out_channels=out_channels,
                num_heads=self.num_heads,
                eps=eps,
                init_zero=init_zero,
                init_attn=init_attn,
                init=init,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
                fused_conv_bias=fused_conv_bias,
            )
        else:
            self.attn = None

    def forward(self, x, emb):
        with (
            _nvtx_noop(message="UNetBlock", color="purple")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            orig = x
            x = self.conv0(self.norm0(x))
            params = self.affine(emb).unsqueeze(2).unsqueeze(3)
            _validate_amp(self.amp_mode)
            if not self.amp_mode:
                if params.dtype != x.dtype:
                    params = params.to(x.dtype)  # type: ignore

            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = self.norm1(x.add_(params))

            x = self.conv1(
                torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            )
            x = x.add_(self.skip(orig) if self.skip is not None else orig)
            x = x * self.skip_scale

            if self.attn:
                x = self.attn(x)
                x = x * self.skip_scale
            return x

    def __setattr__(self, name, value):
        """Prevent setting attributes with reserved names.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        if name in getattr(self.__class__, "_reserved_attributes", set()):
            raise AttributeError(f"Attribute '{name}' is reserved and cannot be set.")
        super().__setattr__(name, value)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Custom ``load_state_dict`` that migrates legacy keys to their
        new locations before loading the state dict.

        Parameters
        ----------
        state_dict : dict
            A state-dict containing parameters and persistent buffers.
        strict : bool, optional
            Passed through to ``torch.nn.Module.load_state_dict``.
        """
        # Migrate legacy keys for the attention module
        self._migrate_attention_module(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def _migrate_attention_module(self, state_dict):
        """Handle legacy checkpoints that stored attention layers at root.

        The earliest versions of ``UNetBlock`` stored the attention-layer
        parameters directly on the block using attribute names contained in
        ``_reserved_attributes``.  These have since been moved under the
        dedicated ``attn`` sub-module.  This helper migrates the parameter
        names so that older checkpoints can still be loaded.
        """

        _mapping = {
            "norm2.weight": "attn.norm.weight",
            "norm2.bias": "attn.norm.bias",
            "qkv.weight": "attn.qkv.weight",
            "qkv.bias": "attn.qkv.bias",
            "proj.weight": "attn.proj.weight",
            "proj.bias": "attn.proj.bias",
        }

        # Track which legacy keys were found.
        legacy_found = set()

        for old_key, new_key in _mapping.items():
            if old_key in state_dict:
                legacy_found.add(old_key)
                # NOTE: Only migrate if destination key not already present to
                # avoid accidental overwriting when both are present.
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
                else:
                    raise ValueError(
                        f"Checkpoint contains both legacy and new keys for {old_key}"
                    )

        # Validation and warnings
        src_keys = set(state_dict.keys()) & set(_mapping.keys())
        target_keys = set(self.state_dict().keys()) & set(_mapping.values())
        missing_keys, unexpected_keys = src_keys - target_keys, target_keys - src_keys
        if missing_keys:
            warnings.warn(
                "The following keys from the checkpoint were not found in the current "
                "model and were ignored: "
                f"{', '.join(sorted(missing_keys))}",
                UserWarning,
            )
        if unexpected_keys:
            warnings.warn(
                "The following keys of the current model are not present in the loaded "
                "checkpoint: "
                f"{', '.join(sorted(unexpected_keys))}",
                UserWarning,
            )


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.

    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10000,
        endpoint: bool = False,
        amp_mode: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.amp_mode = amp_mode

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if freqs.dtype != x.dtype:
                freqs = freqs.to(x.dtype)
        x = x.ger(freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(self, num_channels: int, scale: int = 16, amp_mode: bool = False):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)
        self.amp_mode = amp_mode

    def forward(self, x):
        freqs = self.freqs
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if x.dtype != self.freqs.dtype:
                freqs = self.freqs.to(x.dtype)

        x = x.ger((2 * np.pi * freqs))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class SongUNet(torch.nn.Module):
    r"""
    This architecture is a diffusion backbone for 2D image generation.
    It is a reimplementation of the `DDPM++
    <https://proceedings.mlr.press/v139/nichol21a.html>`_ and `NCSN++ <https://arxiv.org/abs/2011.13456>`_
    architectures, which are U-Net variants
    with optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    This architecture supports conditioning on the noise level (called *noise labels*),
    as well as on additional vector-valued labels (called *class labels*) and (optional)
    vector-valued augmentation labels. The conditioning mechanism relies on addition
    of the conditioning embeddings in the U-Net blocks of the encoder. To condition
    on images, the simplest mechanism is to concatenate the image to the input
    before passing it to the SongUNet.

    The model first applies a mapping operation to generate embeddings for all
    the conditioning inputs (the noise level, the class labels, and the
    optional augmentation labels).

    Then, at each level in the U-Net encoder, a sequence of blocks is applied:

    • A first block downsamples the feature map resolution by a factor of 2
      (odd resolutions are floored). This block does not change the number of
      channels.

    • A sequence of ``num_blocks`` U-Net blocks are applied, each with a different
      number of channels. These blocks do not change the feature map
      resolution, but they multiply the number of channels by a factor
      specified in ``channel_mult``.
      If required, the U-Net blocks also apply self-attention at the specified
      resolutions.

    • At the end of the level, the feature map is cached to be used in a skip
      connection in the decoder.

    The decoder is a mirror of the encoder, with the same number of levels and
    the same number of blocks per level. It multiplies the feature map resolution
    by a factor of 2 at each level.

    Parameters
    -----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int :math:`H` for
        square images or a list :math:`[H, W]` for rectangular images.

        *Note:* This parameter is only used as a convenience to build the
        network. In practice, the model can still be used with images of
        different resolutions. The only exception to this rule is when
        ``additive_pos_embed`` is True, in which case the resolution of the latent
        state :math:`\mathbf{x}` must match ``img_resolution``.
    in_channels : int
        Number of channels :math:`C_{in}` in the input image. May include channels from both
        the latent state and additional channels when conditioning on images.
        For an unconditional model, this should be equal to ``out_channels``.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image. Should be equal to the number
        of channels :math:`C_{\mathbf{x}}` in the latent state.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning; 0
        indicates no conditioning on class labels.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued `augment_labels` conditioning; 0 means
        no conditioning on augmentation labels.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels accross the entire network.
    channel_mult : List[int], optional, default=[1, 2, 2, 2]
        Multipliers for the number of channels at every level in
        the encoder and decoder. The length of ``channel_mult`` determines the
        number of levels in the U-Net. At level ``i``, the number of channel in
        the feature map is ``channel_mult[i] * model_channels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector. The
        embedding vector has ``model_channels * channel_mult_emb`` channels.
    num_blocks : int, optional, default=4
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions of the levels at which self-attention layers are applied.
        Note that the feature map resolution must match exactly the value
        provided in `attn_resolutions` for the self-attention layers to be
        applied.
    dropout : float, optional, default=0.10
        Dropout probability applied to intermediate activations within the
        U-Net blocks.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the `class_labels`. Typically used for
        classifier-free guidance.
    embedding_type : Literal["fourier", "positional", "zero"], optional, default="positional"
        Diffusion timestep embedding type: 'positional' for DDPM++, 'fourier'
        for NCSN++, 'zero' for none.
    channel_mult_noise : int, optional, default=1
        Multiplier for the number of channels in the noise level embedding. The
        noise level embedding vector has ``model_channels * channel_mult_noise`` channels.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture: 'standard' or 'skip' for skip connections.
    resample_filter : List[int], optional, default=[1, 1]
        Resampling filter coefficients applied in the U-Net blocks
        convolutions: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    checkpoint_level : int, optional, default=0
        Number of levels that should use gradient checkpointing. Only levels at
        which the feature map resolution is large enough will be checkpointed
        (0 disables checkpointing, higher values means more layers are checkpointed).
        Higher values trade memory for computation.
    additive_pos_embed : bool, optional, default=False
        If ``True``, adds a learnable positional embedding after the first convolution layer.
        Used in StormCast model.

        *Note:* Those positional embeddings encode spatial position information
        of the image pixels, unlike the ``embedding_type`` parameter which encodes
        temporal information about the diffusion process. In that sense it is a
        simpler version of the positional embedding used in
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`.
    use_apex_gn : bool, optional, default=False
        A flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Apex needs to be installed for this to work. Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
        Required when ``use_apex_gn`` is ``True``.
    profile_mode : bool, optional, default=False
        A flag indicating whether to enable all nvtx annotations during
        profiling.
    amp_mode : bool, optional, default=False
        A flag indicating whether mixed-precision (AMP) training is enabled.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`. In
        general ``x`` is the channel-wise concatenation of the latent state
        :math:`\mathbf{x}` and additional images used for conditioning. For an
        unconditional model, ``x`` is simply the latent state
        :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match
        :math:`H` and :math:`W` defined in ``img_resolution``, except when
        ``additive_pos_embed`` is ``True``. In that case, the resolution of
        ``x`` must match ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        • The terms *noise levels* (or *noise labels*) are used to refer to the diffusion time-step, as these are conceptually equivalent.
        • The terms *labels* and *classes* originate from the original paper and EDM repository,
          where this architecture was used for class-conditional image generation. While these terms
          suggest class-based conditioning, the architecture can actually be conditioned on any vector-valued
          conditioning.
        • The term *positional embedding* used in the `embedding_type` parameter
          also comes from the original paper and EDM repository. Here,
          *positional* refers to the diffusion time-step, similar to how position is used in transformer
          architectures. Despite the name, these embeddings encode temporal information about the
          diffusion process rather than spatial position information.
        • Limitations on input image resolution: for a model that has :math:`N` levels,
          the latent state :math:`\mathbf{x}` must have resolution that is a multiple of :math:`2^N` in each dimension.
          This is due to a limitation in the decoder that does not support shape mismatch
          in the residual connections from the encoder to the decoder. For images that do not match
          this requirement, it is recommended to interpolate your data on a grid of the required resolution
          beforehand.

    Example
    --------
    >>> model = SongUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method.
    _overridable_args: Set[str] = {"use_apex_gn", "act"}

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: Literal["fourier", "positional", "zero"] = "positional",
        channel_mult_noise: int = 1,
        encoder_type: Literal["standard", "skip", "residual"] = "standard",
        decoder_type: Literal["standard", "skip"] = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(
                f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}."
            )

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        super().__init__()
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
            use_apex_gn=use_apex_gn,
            act=act,
            fused_conv_bias=True,
            profile_mode=profile_mode,
            amp_mode=amp_mode,
        )
        self.use_apex_gn = use_apex_gn

        # for compatibility with older versions that took only 1 dimension
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        # set the threshold for checkpointing based on image resolution
        self.checkpoint_threshold = (
            math.floor(math.sqrt(self.img_shape_x * self.img_shape_y))
            >> checkpoint_level
        ) + 1

        # Optional additive learned positition embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(
                torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x)
            )
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(
                    num_channels=noise_channels, endpoint=True, amp_mode=amp_mode
                )
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels, amp_mode=amp_mode)
            )
            self.map_label = (
                Linear(
                    in_features=label_dim,
                    out_features=noise_channels,
                    amp_mode=amp_mode,
                    **init,
                )
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    amp_mode=amp_mode,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels,
                out_features=emb_channels,
                amp_mode=amp_mode,
                **init,
            )
            self.map_layer1 = Linear(
                in_features=emb_channels,
                out_features=emb_channels,
                amp_mode=amp_mode,
                **init,
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    fused_conv_bias=True,
                    amp_mode=amp_mode,
                    **init,
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                        amp_mode=amp_mode,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=1,
                        fused_conv_bias=True,
                        amp_mode=amp_mode,
                        **init,
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        fused_conv_bias=True,
                        amp_mode=amp_mode,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                        amp_mode=amp_mode,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout,
                    eps=1e-6,
                    use_apex_gn=use_apex_gn,
                    amp_mode=amp_mode,
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    fused_conv_bias=True,
                    amp_mode=amp_mode,
                    **init_zero,
                )

        # Set properties recursively on submodules
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode

    # Properties that are recursively set on submodules
    profile_mode = _recursive_property(
        "profile_mode", bool, "Should be set to ``True`` to enable profiling."
    )
    amp_mode = _recursive_property(
        "amp_mode",
        bool,
        "Should be set to ``True`` to enable automatic mixed precision.",
    )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        with (
            _nvtx_noop(message="SongUNet", color="blue")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            if (
                self.use_apex_gn
                and (not x.is_contiguous(memory_format=torch.channels_last))
                and x.dim() == 4
            ):
                x = x.to(memory_format=torch.channels_last)
            if self.embedding_type != "zero":
                # Mapping.
                emb = self.map_noise(noise_labels)
                emb = (
                    emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
                )  # swap sin/cos
                if self.map_label is not None:
                    tmp = class_labels
                    if self.training and self.label_dropout:
                        tmp = tmp * (
                            torch.rand([x.shape[0], 1], device=x.device)
                            >= self.label_dropout
                        ).to(tmp.dtype)
                    emb = emb + self.map_label(
                        tmp * np.sqrt(self.map_label.in_features)
                    )
                if self.map_augment is not None and augment_labels is not None:
                    emb = emb + self.map_augment(augment_labels)
                emb = silu(self.map_layer0(emb))
                emb = silu(self.map_layer1(emb))
            else:
                emb = torch.zeros(
                    (noise_labels.shape[0], self.emb_channels),
                    device=x.device,
                    dtype=x.dtype,
                )

            # Encoder.
            skips = []
            aux = x
            for name, block in self.enc.items():
                with (
                    _nvtx_noop(f"SongUNet encoder: {name}", color="blue")
                    if self.profile_mode
                    else contextlib.nullcontext()
                ):
                    if "aux_down" in name:
                        aux = block(aux)
                    elif "aux_skip" in name:
                        x = skips[-1] = x + block(aux)
                    elif "aux_residual" in name:
                        x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
                    elif "_conv" in name:
                        x = block(x)
                        if self.additive_pos_embed:
                            x = x + self.spatial_emb.to(dtype=x.dtype)
                        skips.append(x)
                    else:
                        # For UNetBlocks check if we should use gradient checkpointing
                        if isinstance(block, UNetBlock):
                            if (
                                math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                                > self.checkpoint_threshold
                            ):
                                # self.checkpoint = checkpoint?
                                # else: self.checkpoint  = lambda(block,x,emb:block(x,emb))
                                x = checkpoint(block, x, emb, use_reentrant=False)
                            else:
                                # AssertionError: Only support NHWC layout.
                                x = block(x, emb)
                        else:
                            x = block(x)
                        skips.append(x)

            # Decoder.
            aux = None
            tmp = None
            for name, block in self.dec.items():
                with (
                    _nvtx_noop(f"SongUNet decoder: {name}", color="blue")
                    if self.profile_mode
                    else contextlib.nullcontext()
                ):
                    if "aux_up" in name:
                        aux = block(aux)
                    elif "aux_norm" in name:
                        tmp = block(x)
                    elif "aux_conv" in name:
                        tmp = block(silu(tmp))
                        aux = tmp if aux is None else tmp + aux
                    else:
                        if x.shape[1] != block.in_channels:
                            x = torch.cat([x, skips.pop()], dim=1)
                        # check for checkpointing on decoder blocks and up sampling blocks
                        if (
                            math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                            > self.checkpoint_threshold
                            and "_block" in name
                        ) or (
                            math.floor(math.sqrt(x.shape[-2] * x.shape[-1]))
                            > (self.checkpoint_threshold / 2)
                            and "_up" in name
                        ):
                            x = checkpoint(block, x, emb, use_reentrant=False)
                        else:
                            x = block(x, emb)
            return aux


# ------------------------------------------------------------------------------
# Specialized architectures
# ------------------------------------------------------------------------------


class SongUNetPosEmbd(SongUNet):
    r"""This specialized architecture extends
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNet` with positional
    embeddings that encode global spatial coordinates of the pixels.

    This model supports the same type of conditioning as the base SongUNet, and
    can be in addition conditioned on the positional embeddings. Conditioning on
    the positional embeddings is performed with a channel-wise concatenation to
    the input image before the first layer of the U-Net. Multiple types of
    positional embeddings are supported. Positional embeddings are represented by
    a 2D grid of shape :math:`(C_{PE}, H, W)`, where :math:`H` and
    :math:`W` correspond to the ``img_resolution`` parameter.

    The following types of positional embeddings are
    supported:

    • learnable: uses a 2D grid of learnable parameters.

    • linear: uses a 2D rectilinear grid over the domain :math:`[-1, 1] \times
      [-1, 1]`.

    • sinusoidal: uses sinusoidal functions of the spatial coordinates, with
      possibly multiple frequency bands.

    • test: uses a 2D grid of integer indices, only used for testing.

    When the input image spatial resolution is smaller than the global
    positional embeddings, it is necessary to select a subset (or *patch*) of the embedding
    grid that correspond to the spatial locations of the input image pixels. The
    model provides two methods for selecting the subset of positional
    embeddings:

    1. Using a selector function. See :meth:`positional_embedding_selector` for
       details.

    2. Using global indices. See :meth:`positional_embedding_indexing` for
       details.

    If none of these are provided, the entire grid of positional embeddings is
    used and channel-wise concatenated to the input image.

    Most parameters are the same as in the parent class
    :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`. Only the ones
    that differ are listed below.

    Parameters
    ----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int for
        square images or a list :math:`[H, W]` for rectangular images.
        Used to set the resolution of the positional embedding grid. It must
        correspond to the spatial resolution of the *global* domain/image.
    in_channels : int
        Number of channels :math:`C_{in} + C_{PE}`, where :math:`C_{in}` is the
        number of channels in the image passed to the U-Net and :math:`C_{PE}`
        is the number of channels in the positional embedding grid.

        **Important:** in comparison to the base
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`, this
        parameter should also include the number of channels in the positional
        embedding grid :math:`C_{PE}`.
    gridtype : Literal["sinusoidal", "learnable", "linear", "test"], optional, default="sinusoidal"
        Type of positional embedding to use. Controls how spatial pixels locations are encoded.
    N_grid_channels : int, optional, default=4
        Number of channels :math:`C_{PE}` in the positional embedding grid. For 'sinusoidal' must be 4 or
        multiple of 4. For 'linear' and 'test' must be 2. For 'learnable' can be any
        value.
    lead_time_mode : bool, optional, default=False
        Provided for convenience. It is recommended to use the architecture
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`
        for a lead-time aware model.
    lead_time_channels : int, optional, default=None
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.
    lead_time_steps : int, optional, default=9
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.
    prob_channels : List[int], optional, default=[]
        Provided for convenience. Refer to :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`,
        where :math:`H_{in}` and :math:`W_{in}` are the spatial dimensions of
        the input image (does not need to be the full image).
        In general ``x`` is the channel-wise concatenation of the latent state :math:`\mathbf{x}` and
        additional images used for conditioning. For an unconditional model, ``x``
        is simply the latent state :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match the
        ``img_resolution`` parameter, except when ``additive_pos_embed`` is ``True``.
        In all other cases, the resolution of ``x`` must be smaller than
        ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    global_index : torch.Tensor, optional, default=None
        The global indices of the positional embeddings to use. If neither
        ``global_index`` nor ``embedding_selector`` are provided, the entire
        positional embedding grid of shape :math:`(C_{PE}, H, W)` is used.
        In this case ``x`` must have the same spatial resolution as the positional
        embedding grid. See :meth:`positional_embedding_indexing` for details.
    embedding_selector : Callable, optional, default=None
        A function that selects the positional embeddings to use. See
        :meth:`positional_embedding_selector` for details.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        Unlike positional embeddings defined by ``embedding_type`` in the parent
        class :class:`~physicsnemo.models.diffusion.song_unet.SongUNet` that
        encode the diffusion time-step (or noise level), the positional embeddings in this
        specialized architecture encode global spatial coordinates of the
        pixels.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.diffusion.song_unet import SongUNetPosEmbd
    >>> from physicsnemo.utils.patching import GridPatching2D
    >>>
    >>> # Model initialization - in_channels must include both original input channels (2)
    >>> # and the positional embedding channels (N_grid_channels=4 by default)
    >>> model = SongUNetPosEmbd(img_resolution=16, in_channels=2+4, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> # The input has only the original 2 channels - positional embeddings are
    >>> # added automatically inside the forward method
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    >>>
    >>> # Using a global index to select all positional embeddings
    >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(16, 16))
    >>> global_index = patching.global_index(batch_size=1)
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     global_index=global_index
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    >>>
    >>> # Using a custom embedding selector to select all positional embeddings
    >>> def patch_embedding_selector(emb):
    ...     return patching.apply(emb[None].expand(1, -1, -1, -1))
    >>> output_image = model(
    ...     input_image, noise_labels, class_labels,
    ...     embedding_selector=patch_embedding_selector
    ... )
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [28],
        dropout: float = 0.13,
        label_dropout: float = 0.0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: List[int] = [1, 1],
        gridtype: Literal["sinusoidal", "learnable", "linear", "test"] = "sinusoidal",
        N_grid_channels: int = 4,
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
        lead_time_mode: bool = False,
        lead_time_channels: int = None,
        lead_time_steps: int = 9,
        prob_channels: List[int] = [],
    ):
        super().__init__(
            img_resolution,
            in_channels,
            out_channels,
            label_dim,
            augment_dim,
            model_channels,
            channel_mult,
            channel_mult_emb,
            num_blocks,
            attn_resolutions,
            dropout,
            label_dropout,
            embedding_type,
            channel_mult_noise,
            encoder_type,
            decoder_type,
            resample_filter,
            checkpoint_level,
            additive_pos_embed,
            use_apex_gn,
            act,
            profile_mode,
            amp_mode,
        )

        self.gridtype = gridtype
        self.N_grid_channels = N_grid_channels
        if self.gridtype == "learnable":
            self.pos_embd = self._get_positional_embedding()
        else:
            self.register_buffer("pos_embd", self._get_positional_embedding().float())
        self.lead_time_mode = lead_time_mode
        if self.lead_time_mode:
            self.lead_time_channels = lead_time_channels
            self.lead_time_steps = lead_time_steps
            self.lt_embd = self._get_lead_time_embedding()
            self.prob_channels = prob_channels
            if self.prob_channels:
                self.scalar = torch.nn.Parameter(
                    torch.ones((1, len(self.prob_channels), 1, 1))
                )

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        global_index: Optional[torch.Tensor] = None,
        embedding_selector: Optional[Callable] = None,
        augment_labels=None,
        lead_time_label=None,
    ):
        with (
            _nvtx_noop(message="SongUNetPosEmbd", color="blue")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            if embedding_selector is not None and global_index is not None:
                raise ValueError(
                    "Cannot provide both embedding_selector and global_index."
                )

            # Append positional embedding to input conditioning
            if self.pos_embd is not None:
                # Select positional embeddings with a selector function
                if embedding_selector is not None:
                    selected_pos_embd = self.positional_embedding_selector(
                        x, embedding_selector, lead_time_label=lead_time_label
                    )
                # Select positional embeddings using global indices (selects all
                # embeddings if global_index is None)
                else:
                    selected_pos_embd = self.positional_embedding_indexing(
                        x, global_index=global_index, lead_time_label=lead_time_label
                    )
                x = torch.cat((x, selected_pos_embd.to(x.dtype)), dim=1)

            out = super().forward(x, noise_labels, class_labels, augment_labels)

            if self.lead_time_mode and self.prob_channels:
                # if training mode, let crossEntropyLoss do softmax. The model outputs logits.
                # if eval mode, the model outputs probability
                scalar = self.scalar
                if out.dtype != scalar.dtype:
                    scalar = scalar.to(out.dtype)
                if self.training:
                    out[:, self.prob_channels] = out[:, self.prob_channels] * scalar
                else:
                    out[:, self.prob_channels] = (
                        (out[:, self.prob_channels] * scalar)
                        .softmax(dim=1)
                        .to(out.dtype)
                    )
            return out

    def positional_embedding_indexing(
        self,
        x: torch.Tensor,
        global_index: Optional[torch.Tensor] = None,
        lead_time_label=None,
    ) -> torch.Tensor:
        r"""Select positional embeddings using global indices.

        This method uses global indices to select specific subset of the
        positional embedding grid (called *patches*). If no indices are provided,
        the entire positional embedding grid is returned.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(P \times B, C, H_{in}, W_{in})`.
            Only used to determine batch size :math:`B` and device.
        global_index : Optional[torch.Tensor], default=None
            Tensor of shape :math:`(P, 2, H_{in}, W_{in})` that correspond to
            the patches to extract from the positional embedding grid.
            :math:`P` is the number of distinct patches in the input tensor ``x``.
            The channel dimension should contain :math:`j`, :math:`i` indices that
            should represent the indices of the pixels to extract from the embedding grid.

        Returns
        -------
        torch.Tensor
            Selected positional embeddings with shape :math:`(P \times B, C_{PE}, H_{in}, W_{in})`
            (same spatial resolution as ``global_index``) if ``global_index`` is provided.
            If ``global_index`` is None, the entire positional embedding grid
            is duplicated :math:`B` times and returned with shape :math:`(B, C_{PE}, H, W)`.

        Example
        -------
        >>> # Create global indices using patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> global_index = patching.global_index(batch_size=3)
        >>> print(global_index.shape)
        torch.Size([4, 2, 8, 8])

        Notes
        -----
            - This method is typically used in patch-based diffusion (or
              multi-diffusion), where a large input image is split into
              multiple patches. The batch dimension of the input tensor contains the patches.
              Patches are processed independently by the model, and the ``global_index`` parameter
              is used to select the grid of positional embeddings corresponding
              to each patch.
            - See this method from :class:`physicsnemo.utils.patching.BasePatching2D`
              for generating the ``global_index`` parameter:
              :meth:`~physicsnemo.utils.patching.BasePatching2D.global_index`.
        """
        # If no global indices are provided, select all embeddings and expand
        # to match the batch size of the input
        pos_embd = self.pos_embd
        if x.dtype != pos_embd.dtype:
            pos_embd = pos_embd.to(x.dtype)

        if global_index is None:
            if self.lead_time_mode:
                selected_pos_embd = []
                if pos_embd is not None:
                    selected_pos_embd.append(
                        pos_embd[None].expand((x.shape[0], -1, -1, -1))
                    )
                if self.lt_embd is not None:
                    selected_pos_embd.append(
                        torch.reshape(
                            self.lt_embd[lead_time_label.int()],
                            (
                                x.shape[0],
                                self.lead_time_channels,
                                self.img_shape_y,
                                self.img_shape_x,
                            ),
                        )
                    )
                if len(selected_pos_embd) > 0:
                    selected_pos_embd = torch.cat(selected_pos_embd, dim=1)
            else:
                selected_pos_embd = pos_embd[None].expand(
                    (x.shape[0], -1, -1, -1)
                )  # (B, C_{PE}, H, W)

        else:
            P = global_index.shape[0]
            B = x.shape[0] // P
            H = global_index.shape[2]
            W = global_index.shape[3]

            global_index = torch.reshape(
                torch.permute(global_index, (1, 0, 2, 3)), (2, -1)
            )  # (P, 2, X, Y) to (2, P*X*Y)
            selected_pos_embd = pos_embd[
                :, global_index[0], global_index[1]
            ]  # (C_pe, P*X*Y)
            selected_pos_embd = torch.permute(
                torch.reshape(selected_pos_embd, (pos_embd.shape[0], P, H, W)),
                (1, 0, 2, 3),
            )  # (P, C_pe, X, Y)

            selected_pos_embd = selected_pos_embd.repeat(
                B, 1, 1, 1
            )  # (B*P, C_pe, X, Y)

            # Append positional and lead time embeddings to input conditioning
            if self.lead_time_mode:
                embeds = []
                if pos_embd is not None:
                    embeds.append(selected_pos_embd)  # reuse code below
                if self.lt_embd is not None:
                    lt_embds = self.lt_embd[
                        lead_time_label.int()
                    ]  # (B, self.lead_time_channels, self.img_shape_y, self.img_shape_x),

                    selected_lt_pos_embd = lt_embds[
                        :, :, global_index[0], global_index[1]
                    ]  # (B, C_lt, P*X*Y)
                    selected_lt_pos_embd = torch.reshape(
                        torch.permute(
                            torch.reshape(
                                selected_lt_pos_embd,
                                (B, self.lead_time_channels, P, H, W),
                            ),
                            (0, 2, 1, 3, 4),
                        ).contiguous(),
                        (B * P, self.lead_time_channels, H, W),
                    )  # (B*P, C_pe, X, Y)
                    embeds.append(selected_lt_pos_embd)

                if len(embeds) > 0:
                    selected_pos_embd = torch.cat(embeds, dim=1)

        return selected_pos_embd

    def positional_embedding_selector(
        self,
        x: torch.Tensor,
        embedding_selector: Callable[[torch.Tensor], torch.Tensor],
        lead_time_label=None,
    ) -> torch.Tensor:
        r"""Select positional embeddings using a selector function.

        Similar to :meth:`positional_embedding_indexing`, but instead uses a selector
        function to select the embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(P \times B, C, H_{in}, W_{in})`.
            Only used to determine batch size :math:`B`, dtype and device.
        embedding_selector : Callable
            Function that takes as input the entire embedding grid of shape
            :math:`(C_{PE}, H, W)` and returns selected embeddings with shape
            :math:`(P \times B, C_{PE}, H_{in}, W_{in})`.
            Each selected embedding should correspond to the portion of the embedding grid
            that corresponds to the batch element in ``x``.
            Typically this should be based on
            :meth:`physicsnemo.utils.patching.BasePatching2D.apply` method to
            maintain consistency with patch extraction.
        lead_time_label : Optional[torch.Tensor], default=None
            Tensor of shape :math:`(P,)` that corresponds to the lead-time label for each patch.
            Only used if ``lead_time_mode`` is True.

        Returns
        -------
        torch.Tensor
            A tensor of shape :math:`(P \times B, C_{PE} [+ C_{LT}], H_{in}, W_{in})`.
            :math:`C_{PE}` is the number of embedding channels in the positional embedding grid,
            and :math:`C_{LT}` is the number of embedding channels in the lead-time embedding grid.
            If ``lead_time_label`` is provided, the lead-time embedding channels are included.

        Notes
        -----
            - This method is typically used in patch-based diffusion (or
              multi-diffusion), where a large input image is split into
              multiple patches. The batch dimension of the input tensor contains the patches.
              Patches are processed independently by the model, and the ``embedding_selector`` function is used
              to select the grid of positional embeddings corresponding to each
              patch.
            - See this method from :class:`physicsnemo.utils.patching.BasePatching2D`
              for generating the ``embedding_selector`` parameter:
              :meth:`~physicsnemo.utils.patching.BasePatching2D.apply`

        Example
        -------
        >>> # Define a selector function with a patching utility:
        >>> from physicsnemo.utils.patching import GridPatching2D
        >>> patching = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
        >>> batch_size = 4
        >>> def embedding_selector(emb):
        ...     return patching.apply(emb[None].expand(batch_size, -1, -1, -1))
        >>>
        """
        pos_embd = self.pos_embd
        if x.dtype != pos_embd.dtype:
            pos_embd = pos_embd.to(x.dtype)
        if lead_time_label is not None:
            # TODO: here we assume all patches share same lead_time_label -->
            # need be changed
            embeddings = torch.cat([pos_embd, self.lt_embd[lead_time_label[0].int()]])
        else:
            embeddings = pos_embd
        return embedding_selector(embeddings)  # (B, N_pe, H, W)

    def _get_positional_embedding(self):
        if self.N_grid_channels == 0:
            return None
        elif self.gridtype == "learnable":
            grid = torch.nn.Parameter(
                torch.randn(self.N_grid_channels, self.img_shape_y, self.img_shape_x)
            )  # (N_grid_channels, img_shape_y, img_shape_x)
        elif self.gridtype == "linear":
            if self.N_grid_channels != 2:
                raise ValueError("N_grid_channels must be set to 2 for gridtype linear")
            y = np.meshgrid(np.linspace(-1, 1, self.img_shape_y))
            x = np.meshgrid(np.linspace(-1, 1, self.img_shape_x))
            grid_y, grid_x = np.meshgrid(x, y)
            grid = torch.from_numpy(
                np.stack((grid_y, grid_x), axis=0)
            )  # (2, img_shape_y, img_shape_x)
            grid.requires_grad = False
        elif self.gridtype == "sinusoidal" and self.N_grid_channels == 4:
            # print('sinusuidal grid added ......')
            x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, self.img_shape_x)))
            x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, self.img_shape_x)))
            y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, self.img_shape_y)))
            y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, self.img_shape_y)))
            grid_y1, grid_x1 = np.meshgrid(x1, y1)
            grid_y2, grid_x2 = np.meshgrid(x2, y2)
            grid = torch.from_numpy(
                np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0)
            )
            grid.requires_grad = False
        elif self.gridtype == "sinusoidal" and self.N_grid_channels != 4:
            if self.N_grid_channels % 4 != 0:
                raise ValueError("N_grid_channels must be a factor of 4")
            num_freq = self.N_grid_channels // 4
            freq_bands = 2.0 ** np.linspace(0.0, num_freq, num=num_freq)
            grid_list = []
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, 2 * np.pi, self.img_shape_x),
                np.linspace(0, 2 * np.pi, self.img_shape_y),
            )
            for freq in freq_bands:
                for p_fn in [np.sin, np.cos]:
                    grid_list.append(p_fn(grid_x * freq))
                    grid_list.append(p_fn(grid_y * freq))
            grid = torch.from_numpy(
                np.stack(grid_list, axis=0)
            )  # (N_grid_channels, img_shape_y, img_shape_x)
            grid.requires_grad = False
        elif self.gridtype == "test" and self.N_grid_channels == 2:
            idx_x = torch.arange(self.img_shape_x)
            idx_y = torch.arange(self.img_shape_y)
            mesh_y, mesh_x = torch.meshgrid(idx_y, idx_x)
            grid = torch.stack((mesh_y, mesh_x), dim=0)  # (2, img_shape_y, img_shape_x)
        else:
            raise ValueError("Gridtype not supported.")
        return grid

    def _get_lead_time_embedding(self):
        if (self.lead_time_steps is None) or (self.lead_time_channels is None):
            return None
        grid = torch.nn.Parameter(
            torch.randn(
                self.lead_time_steps,
                self.lead_time_channels,
                self.img_shape_y,
                self.img_shape_x,
            )
        )  # (lead_time_steps, lead_time_channels, img_shape_y, img_shape_x)
        return grid


# TODO: the entire logic of the lead-time logic should be moved there. We
# should use subclass of the SongUNetPosEmbd class and specialize it for
# lead-time aware embeddings.


def _wrapped_property(prop_name: str, wrapped_obj_name: str, doc: str) -> property:
    """
    Property factory to define a property on a Module ``self`` that is
    wraps another Module in an attribute ``self.<wrapped_obj_name>``. The
    property delegates the setter and getter to the wrapped object's.

    Parameters
    ----------
    prop_name : str
        The name of the property.
    wrapped_obj_name : str
        The name of the attribute that wraps the other Module.
    doc : str
        The documentation string for the property.

    Returns
    -------
    property
        The property object.
    """

    def _setter(self, value: Any):
        wrapped_obj = getattr(self, wrapped_obj_name)
        if hasattr(wrapped_obj, prop_name):
            setattr(wrapped_obj, prop_name, value)
        else:
            raise AttributeError(f"{prop_name} is not supported by the wrapped model.")

    def _getter(self):
        wrapped_obj = getattr(self, wrapped_obj_name)
        return getattr(wrapped_obj, prop_name)

    return property(_getter, _setter, doc=doc)


class UNet(torch.nn.Module):
    r"""
    This interface provides a U-Net wrapper for CorrDiff deterministic
    regression model (and other deterministic downsampling models).
    It supports the following architectures:

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`

    - :class:`~physicsnemo.models.diffusion.dhariwal_unet.DhariwalUNet`

    It shares the same architeture as a conditional diffusion model. It does so
    by concatenating a conditioning image to a zero-filled latent state, and by
    setting the noise level and the class labels to zero.

    Parameters
    -----------
    img_resolution : Union[int, Tuple[int, int]]
        The resolution of the input/output image. If a single int is provided,
        then the image is assumed to be square.
    img_in_channels : int
        Number of channels in the input image.
    img_out_channels : int
        Number of channels in the output image.
    use_fp16: bool, optional, default=False
        Execute the underlying model at FP16 precision.
    model_type: Literal['SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd',
    'DhariwalUNet'], default='SongUNetPosEmbd'
        Class name of the underlying architecture. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
    **model_kwargs : dict
        Keyword arguments passed to the underlying architecture `__init__` method.

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Forward
    -------
    x : torch.Tensor
        The input tensor, typically zero-filled, of shape :math:`(B, C_{in}, H_{in}, W_{in})`.
    img_lr : torch.Tensor
        Conditioning image of shape :math:`(B, C_{lr}, H_{in}, W_{in})`.
    **model_kwargs : dict
        Additional keyword arguments to pass to the underlying architecture
        forward method.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})` (same
        spatial dimensions as the input).

    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": "Loading UNet checkpoint from older version 0.1.0 (current version is 0.2.0). This version is still supported, but consider re-saving the model to upgrade to version 0.2.0 and remove this warning."
    }

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes: Set[str] = {
        "SongUNetPosEmbd",
        "SongUNet",
    }

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method. Here, since we use splatted arguments
    # for the wrapped model instance, we allow overriding of any overridable
    # argument of the wrapped classes.
    _overridable_args: Set[str] = set.union(
        *(
            getattr(globals()[cls_name], "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map arguments from older versions to current version format.

        Parameters
        ----------
        version : str
            Version of the checkpoint being loaded
        args : Dict[str, Any]
            Arguments dictionary from the checkpoint

        Returns
        -------
        Dict[str, Any]
            Updated arguments dictionary compatible with current version
        """
        # Call parent class method first
        args = super()._backward_compat_arg_mapper(version, args)

        if version == "0.1.0":
            # In version 0.1.0, img_channels was unused
            if "img_channels" in args:
                _ = args.pop("img_channels")

            # Sigma parameters are also unused
            if "sigma_min" in args:
                _ = args.pop("sigma_min")
            if "sigma_max" in args:
                _ = args.pop("sigma_max")
            if "sigma_data" in args:
                _ = args.pop("sigma_data")

        return args

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: dict,
    ):
        super().__init__()

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        # for compatibility with older versions that took only 1 dimension
        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        model_class = globals()[model_type]
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )
        self.use_fp16 = use_fp16

    # Properties delegated to the wrapped model
    amp_mode = _wrapped_property(
        "amp_mode",
        "model",
        "Set to ``True`` when using automatic mixed precision.",
    )
    profile_mode = _wrapped_property(
        "profile_mode",
        "model",
        "Set to ``True`` to enable profiling of the wrapped model.",
    )

    @property
    def use_fp16(self):
        """
        bool: Whether the model uses float16 precision.

        Returns
        -------
        bool
            True if the model is in float16 mode, False otherwise.
        """
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        """
        Set whether the model should use float16 precision.

        Parameters
        ----------
        value : bool
            If True, moves the model to torch.float16. If False, moves to torch.float32.

        Raises
        ------
        ValueError
            If `value` is not a boolean.
        """
        # NOTE: allow 0/1 values for older checkpoints
        if not (isinstance(value, bool) or value in [0, 1]):
            raise ValueError(
                f"`use_fp16` must be a boolean, but got {type(value).__name__}."
            )
        self._use_fp16 = value
        if value:
            self.to(torch.float16)
        else:
            self.to(torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        # SR: concatenate input channels
        if img_lr is not None:
            x = torch.cat((x, img_lr), dim=1)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self.model(
            x.to(dtype),  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], dtype=dtype, device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        # skip connection
        D_x = F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


# TODO: implement amp_mode and profile_mode properties for StormCastUNet (same
# as UNet)
