#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 源自 NVIDIA PhysicsNeMo (Apache-2.0), release v1.2.0,
# commit 17efe3e3f4d179fc59e822543697add589426147, physicsnemo/metrics/diffusion/loss.py
"""
============================================================================
corrdiff_loss.py — CorrDiff 阶段 B 的残差扩散损失
============================================================================
两阶段残差分解的训练目标: 先用冻结的回归网得到全域均值 mu, 在全域上形成残差 y - mu,
再切成 patch 交给扩散网去噪。切块发生在残差形成之后, 因此 mu 与 patch 位置无关。

从官方源码移植, 数学部分未改; 仅把 patching 的导入指向本仓库的同名移植模块。
============================================================================
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from era5_daymet.models.patching import RandomPatching2D


class ResidualLoss:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching. It uses a pre-trained regression
    network to compute residuals before applying the diffusion process.

    Parameters
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise
            concatenated with `img_lr` for conditioning.
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None

    def get_noise_params(self, y: Tensor) -> Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y) * sigma
        return n, sigma, weight

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[
            Callable[[Tensor], Tuple[Tensor, Optional[Tensor]]]
        ] = None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        Calculate and return the loss for denoising score matching.

        This method computes a mixture loss that combines deterministic
        regression with denoising score matching. It first computes residuals
        using the regression network, then applies the diffusion process to
        these residuals.

        In addition to the standard denoising score matching loss, this method
        also supports optional patching for multi-diffusion. In this case, the spatial
        dimensions of the input are decomposed into `P` smaller patches of shape
        (H_patch, W_patch), that are grouped along the batch dimension, and the
        model is applied to each patch individually. In the following, if `patching`
        is not provided, then the input is not patched and `P=1` and `(H_patch,
        W_patch) = (H, W)`. When patching is used, the original non-patched conditioning is
        interpolated onto a spatial grid of shape `(H_patch, W_patch)` and channel-wise
        concatenated to the patched conditioning. This ensures that each patch
        maintains global information from the entire domain.

        The diffusion model `net` is expected to be conditioned on an input with
        `C_cond` channels, which should be:
            - `C_cond = C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is None.
            - `C_cond = C_hr + C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is None.
            - `C_cond = C_hr + 2*C_lr` if `hr_mean_conditioning` is `True` and
              `patching` is not None.
            - `C_cond = 2*C_lr` if `hr_mean_conditioning` is `False` and
              `patching` is not None.
        Additionally, `C_cond` should also include any embedding channels,
        such as positional embeddings or time embeddings.

        Note: this loss function does not apply any reduction.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model for the diffusion process.
            Expected signature: `net(latent, y_lr, sigma,
            embedding_selector=embedding_selector, lead_time_label=lead_time_label,
            augment_labels=augment_labels)`, where:
                latent (torch.Tensor): Noisy input of shape (B[*P], C_hr, H_patch, W_patch)
                y_lr (torch.Tensor): Conditioning of shape (B[*P], C_cond, H_patch, W_patch)
                sigma (torch.Tensor): Noise level of shape (B[*P], 1, 1, 1)
                embedding_selector (callable, optional): Function to select
                    positional embeddings. Only used if `patching` is provided.
                lead_time_label (torch.Tensor, optional): Lead time labels.
                augment_labels (torch.Tensor, optional): Augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B[*P], C_hr, H_patch, W_patch)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the regression network and conditioning for the
            diffusion process.

        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None. See
            :class:`physicsnemo.utils.patching.RandomPatching2D` for details.
            When provided, the patching strategy is used for both image patches
            and positional embeddings selection in the diffusion model `net`.
            Transforms tensors from shape (B, C, H, W) to (B*P, C, H_patch,
            W_patch).

        lead_time_label : Optional[torch.Tensor], optional
            Labels for lead-time aware predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.

        augment_pipe : Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]
            Data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution images
                    of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels
        use_patch_grad_acc: bool, optional
            A boolean flag indicating whether to enable multi-iterations of patching accumulations
            for amortizing regression cost. Default False.

        Returns
        -------
        torch.Tensor
            If patching is not used:
                A tensor of shape (B, C_hr, H, W) representing the per-sample loss.
            If patching is used:
                A tensor of shape (B*P, C_hr, H_patch, W_patch) representing
                the per-patch loss.

        Raises
        ------
        ValueError
            If patching is provided but is not an instance of RandomPatching2D.
            If shapes of img_clean and img_lr are incompatible.
        """

        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # if using multi-iterations of patching, switch to optimized version
        if use_patch_grad_acc:
            # form residual
            if self.y_mean is None:
                if lead_time_label is not None:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        lead_time_label=lead_time_label,
                        augment_labels=augment_labels,
                    )
                else:
                    y_mean = self.regression_net(
                        torch.zeros_like(y, device=img_clean.device),
                        y_lr_res,
                        augment_labels=augment_labels,
                    )
                self.y_mean = y_mean

        # if on full domain, or if using patching without multi-iterations
        else:
            # form residual
            if lead_time_label is not None:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
            else:
                y_mean = self.regression_net(
                    torch.zeros_like(y, device=img_clean.device),
                    y_lr_res,
                    augment_labels=augment_labels,
                )

            self.y_mean = y_mean

        y = y - self.y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        # removed patch_embedding_selector due to compilation issue with dynamo.
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)

            y = y_patched
            y_lr = y_lr_patched

        # Add noise to the latent state
        n, sigma, weight = self.get_noise_params(y)

        if lead_time_label is not None:
            D_yn = net(
                y + n,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                y + n,
                y_lr,
                sigma,
                embedding_selector=None,
                global_index=(
                    patching.global_index(batch_size, img_clean.device)
                    if patching is not None
                    else None
                ),
                augment_labels=augment_labels,
            )
        loss = weight * ((D_yn - y) ** 2)

        return loss
