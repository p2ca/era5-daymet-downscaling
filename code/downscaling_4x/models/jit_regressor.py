#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
jit_regressor.py — 确定性回归版 JiT (两阶段的阶段 A)
============================================================================
net(cond, offset) -> μ。与扩散版 `JiT` 的差别只有两处:

  1. 输入只有条件场, 没有加噪目标 —— patch 嵌入宽度是 cond_ch, 不是 cond_ch + out_ch;
  2. 没有时间步 —— adaLN 的条件向量换成一个学习常量, 去掉 TimestepEmbedder。

第 2 点不是为了省那点参数, 是为了**参数量口径的公平**: 保留一个恒定输入的
TimestepEmbedder 会把它的参数计进"同量级下 transformer 对比 UNet"的那个量级里, 而它对
回归没有任何贡献。阶段 A 的全部意义就是这个对比, 口径不能被死参数污染。

注意力块、MoE 挂点、patch 嵌入、位置编码、RoPE、输出头全部直接复用 `jit_backbone` 的实现,
本文件不含任何新的数学。MoE 的配置字典与扩散版逐字段相同。

切块起点 `offset` 仍要逐步重掷, 与是否加噪无关: 固定起点时同一个像素永远落在块内同一
位置, 而逐像素损失从不惩罚"相邻块在边界上对不上", 接缝因此是免费的, 会在固定位置累积成
网格。换起点后同一像素这次在边界、下次在块内, 位置特异的偏置在平均意义上要挨罚。
============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from downscaling_4x.contract import DEFAULT_MODE, cond_channels
from downscaling_4x.models.jit_backbone import (BottleneckPatchEmbed, FinalLayer, JiTBlock,
                                                RefineHead, Rope2D, draw_patch_offset)  # noqa: F401
from downscaling_4x.models.moe_ffn import DSMoE
from downscaling_4x.models.pos_embed import get_2d_sincos_pos_embed


class JiTRegressor(nn.Module):
    """确定性 JiT 回归器。

    cond : (B, cond_ch, H, W) 条件场
    返回 : (B, out_ch, H, W)
    moe_config: None 为全稠密; dict 时按 interleave 把 FFN 换成 DSMoE
                (True: 奇数索引块, 第 0 块恒稠密; False: 全部块)。
    """

    def __init__(self, hw=(480, 960), patch=32, cond_ch=None, out_ch=1,
                 hidden=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 bottleneck=128, attn_drop=0.0, proj_drop=0.0, moe_config=None,
                 patch_margin=0, refine_head=0, refine_static_idx=None):
        super().__init__()
        cond_ch = cond_channels(DEFAULT_MODE) if cond_ch is None else int(cond_ch)
        self.hw, self.patch, self.out_ch = tuple(hw), patch, out_ch
        self.hidden, self.depth = hidden, depth
        # 切块网格补到"不小于 H+patch-1 的最小可整除尺寸": 任意起点 (dy,dx)∈[0,patch)² 都
        # 放得下, 且**形状恒定** —— 位置编码与 RoPE 按固定网格预生成, 形状一变就得重建。
        self.grid_hw = (-(-(self.hw[0] + patch - 1) // patch) * patch,
                        -(-(self.hw[1] + patch - 1) // patch) * patch)
        self.x_embedder = BottleneckPatchEmbed(self.grid_hw, patch, cond_ch,
                                               bottleneck, hidden, margin=patch_margin)
        gh, gw = self.x_embedder.gh, self.x_embedder.gw
        self.register_buffer("pos_embed",
                             get_2d_sincos_pos_embed(hidden, gh, gw).float(),
                             persistent=False)
        self.rope = Rope2D(hidden // num_heads, gh, gw)
        # adaLN 的条件向量: 扩散版由时间步给出, 这里没有时间步, 用一个学习常量代替。
        # 保留 adaLN 而不是拆掉它, 是为了让 block 与扩散版逐字节相同, 两阶段可直接对比。
        self.cond_token = nn.Parameter(torch.zeros(hidden))
        if moe_config is not None:
            use_moe = [(i % 2 == 1) if moe_config.get("interleave", True) else True
                       for i in range(depth)]
        else:
            use_moe = [False] * depth
        mid = lambda i: (depth // 4 * 3 > i >= depth // 4)       # dropout 只在中段块
        self.blocks = nn.ModuleList([
            JiTBlock(hidden, num_heads, mlp_ratio,
                     attn_drop=attn_drop if mid(i) else 0.0,
                     proj_drop=proj_drop if mid(i) else 0.0,
                     moe_config=moe_config if use_moe[i] else None)
            for i in range(depth)])
        self.final_layer = FinalLayer(hidden, patch, out_ch)
        self.refine = None
        self.initialize_weights()
        # 精修头在 initialize_weights **之后**构造: 该函数会再消耗全局随机流(proj1/2 的
        # xavier 重初始化), 头若先构造会移动随机流, 令关/开两侧的主干参数不再逐位相同。
        # 关闭时不构造子模块: 参数数量/遍历顺序/随机数消耗与无此参数的版本逐字节相同,
        # 旧 checkpoint 原样可载, 已跑实验自动成为消融矩阵的"关"格。
        if int(refine_head):
            assert refine_static_idx, "refine_head=1 需要静态通道切片索引"
            self.refine_static_idx = [int(i) for i in refine_static_idx]
            self.refine = RefineHead(out_ch, len(self.refine_static_idx))

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)                       # TopkRouter.weight 是裸 Parameter, 不覆盖
        for w in (self.x_embedder.proj1.weight, self.x_embedder.proj2.weight):
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(B, gh*gw, p*p*C) -> (B, C, gh*p, gw*p), 矩形网格。"""
        B = x.shape[0]
        gh, gw, p, c = self.x_embedder.gh, self.x_embedder.gw, self.patch, self.out_ch
        x = x.reshape(B, gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, c, gh * p, gw * p)

    def forward(self, cond, offset=(0, 0)):
        H, W = self.hw
        Hp, Wp = self.grid_hw
        dy, dx = int(offset[0]) % self.patch, int(offset[1]) % self.patch
        x = F.pad(cond, (dx, Wp - W - dx, dy, Hp - H - dy), mode="reflect")
        x = self.x_embedder(x)
        x = x + self.pos_embed.to(x.dtype)
        c = self.cond_token.to(x.dtype).expand(x.shape[0], -1)
        for blk in self.blocks:
            x = blk(x, c, self.rope)
        out = self.unpatchify(self.final_layer(x, c))
        out = out[..., dy:dy + H, dx:dx + W]
        if self.refine is not None:
            out = self.refine(out, cond[:, self.refine_static_idx])
        return out

    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, DSMoE)]

    def param_counts(self):
        """总参数与"激活参数"(路由专家按 K/E 折算)。"""
        total = sum(p.numel() for p in self.parameters())
        routed = act_routed = 0
        for m in self.moe_layers():
            r = sum(p.numel() for p in m.experts.parameters())
            routed += r
            act_routed += r * m.top_k // m.n_experts
        return {"total": total, "routed_experts": routed,
                "activated": total - routed + act_routed}
