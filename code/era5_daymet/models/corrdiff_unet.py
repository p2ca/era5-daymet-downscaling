#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
corrdiff_unet.py — 按 CorrDiff regression UNet 几何构型的确定性主体
============================================================================
与 `unet.py` 的 `UNet`(2 次下采样、每级 1 个 ResBlock、无注意力)相比, 本模块把**几何构型**
做成可配置的: 层级数由 `channel_mult` 的长度决定, 每级块数由 `num_blocks` 决定, 瓶颈自注意力
可开关。三者组合即可覆盖从"仅加深"到"完整 CorrDiff 构型"的整个区间, 无需改代码。

几何要点(与 CorrDiff 的 regression UNet 对齐):

- **层级**: `channel_mult=(1,2,2,2,2)` 表示 5 级 = 4 次下采样。720x1440 因此降到 45x90 —— 这正是
  瓶颈自注意力能否负担的分水岭: 停在 2 次下采样时瓶颈是 180x360 = 64,800 token, 注意力分数
  达 42 亿/头; 降到 45x90 = 4,050 token 后只剩 1,640 万, 才付得起。
- **skip 记账**: 编码器每一个条目(首层卷积、每次下采样、每个块)都压一个 skip; 解码器每级
  `num_blocks+1` 个块各弹出一个。两侧数量恒等, 与官方实现同构。
- **瓶颈注意力**: 官方实现里瓶颈块的注意力是无条件开启的, 与 `attn_resolutions` 无关
  (那个参数只控制各级**额外**的注意力, 且其判据在 720/192 上永远不命中)。这里把它显式做成
  `attn_bottleneck` 开关, 以便单独消融。
- **解码器对称**: 每级宽度镜像编码器, 不像 `UNet` 那样在上采样后立刻收窄到 base。

块的内部构造沿用本仓库的 `ResBlock`(GroupNorm(8) + 两个 3x3 卷积 + 1x1 skip), 与官方 EDM
UNetBlock 的 skip_scale / 权重初始化 / resample filter 等细节不同; 需要逐比特对齐官方数值时
应直接使用官方类, 本模块负责的是几何构型这一层。
============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from era5_daymet.models.unet import DOMAIN_HW, ResBlock, sincos_pos_grid, sinusoidal_emb


class AttnBlock(nn.Module):
    """瓶颈自注意力。头数按每头 64 通道切分, 与 CorrDiff 的 `out_channels // 64` 同口径。"""

    def __init__(self, ch, channels_per_head=64):
        super().__init__()
        self.heads = max(1, ch // channels_per_head)
        if ch % self.heads:
            raise ValueError(f"通道数 {ch} 无法被头数 {self.heads} 整除")
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        # (B, heads, HW, C/heads) —— SDPA 走 flash 路径, 不 materialize HWxHW
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(a.transpose(-1, -2).reshape(B, C, H, W))


class CorrDiffUNet(nn.Module):
    """几何构型可配置的 UNet。

    参数名沿用 CorrDiff 侧口径(`model_channels` / `channel_mult` / `num_blocks`), 便于与官方
    配置逐字段对照, 不再需要在 `base` 与 `model_channels` 之间做心算换算。

    `pos_grid=4` 时在输入上拼 4 个全域正弦位置通道; 它是模型内部输入, 不改变外部的数据通道
    合同。裁块前向时须传入该块左上角 `origin`, 否则位置口径会与整幅不一致。
    """

    def __init__(self, in_ch, out_ch, model_channels=64, channel_mult=(1, 2, 2, 2, 2),
                 num_blocks=4, attn_bottleneck=True, pos_grid=0, temb=0, domain=DOMAIN_HW):
        super().__init__()
        channel_mult = tuple(int(m) for m in channel_mult)
        if len(channel_mult) < 1:
            raise ValueError("channel_mult 至少要有一级")
        self.channel_mult = channel_mult
        self.num_blocks = int(num_blocks)
        self.temb = temb
        self.pos_grid = int(pos_grid)
        if self.pos_grid not in (0, 4):
            raise ValueError(f"pos_grid 只支持 0(关闭) 或 4(正弦), 收到 {pos_grid}")
        self.domain = (int(domain[0]), int(domain[1]))
        self._pos_cache = {}
        if temb:
            self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))

        # ---- 编码器 ----
        cout = model_channels
        self.inc = nn.Conv2d(in_ch + self.pos_grid, cout, 3, padding=1)
        skip_ch = [cout]                                   # 首层卷积的输出也是一个 skip
        self.enc_down = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            if level == 0:
                self.enc_down.append(nn.Identity())
            else:
                self.enc_down.append(nn.Conv2d(cout, cout, 4, 2, 1))
                skip_ch.append(cout)
            blocks = nn.ModuleList()
            for _ in range(self.num_blocks):
                blocks.append(ResBlock(cout, model_channels * mult, temb))
                cout = model_channels * mult
                skip_ch.append(cout)
            self.enc_blocks.append(blocks)

        # ---- 瓶颈 ----
        self.mid1 = ResBlock(cout, cout, temb)
        self.mid_attn = AttnBlock(cout) if attn_bottleneck else nn.Identity()
        self.mid2 = ResBlock(cout, cout, temb)

        # ---- 解码器(层级顺序与编码器相反) ----
        self.dec_up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            if level == len(channel_mult) - 1:
                self.dec_up.append(nn.Identity())           # 瓶颈级不上采样
            else:
                self.dec_up.append(nn.ConvTranspose2d(cout, cout, 4, 2, 1))
            blocks = nn.ModuleList()
            for _ in range(self.num_blocks + 1):
                blocks.append(ResBlock(cout + skip_ch.pop(), model_channels * mult, temb))
                cout = model_channels * mult
            self.dec_blocks.append(blocks)
        if skip_ch:
            raise RuntimeError(f"skip 记账不平: 解码器用完后仍剩 {len(skip_ch)} 个")

        self.outc = nn.Sequential(nn.GroupNorm(8, cout), nn.SiLU(),
                                  nn.Conv2d(cout, out_ch, 3, padding=1))

    def _pos(self, x, origin):
        key = (x.shape[-2], x.shape[-1], int(origin[0]), int(origin[1]), x.device, x.dtype)
        g = self._pos_cache.get(key)
        if g is None:
            g = sincos_pos_grid(x.shape[-2], x.shape[-1], origin[0], origin[1],
                                self.domain, x.device, x.dtype)
            self._pos_cache = {key: g}
        return g.expand(x.shape[0], -1, -1, -1)

    def forward(self, x, t=None, origin=(0, 0)):
        if self.pos_grid:
            x = torch.cat([x, self._pos(x, origin)], 1)
        te = self.tmlp(sinusoidal_emb(t, self.temb)) if (self.temb and t is not None) else None

        h = self.inc(x)
        skips = [h]
        for down, blocks in zip(self.enc_down, self.enc_blocks):
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)
            for blk in blocks:
                h = blk(h, te)
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, te)), te)

        for up, blocks in zip(self.dec_up, self.dec_blocks):
            if not isinstance(up, nn.Identity):
                h = up(h)
            for blk in blocks:
                h = blk(torch.cat([h, skips.pop()], 1), te)
        return self.outc(h)
